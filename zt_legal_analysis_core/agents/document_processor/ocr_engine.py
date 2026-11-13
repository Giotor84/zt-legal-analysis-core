# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
Document Processor Agent - OCR e Parsing Documenti Multi-formato

Formati supportati:
- PDF (.pdf) - nativi e scansioni
- Immagini (.jpg, .jpeg, .png, .tiff, .bmp)
- Word (.docx, .doc)
- Excel (.xlsx, .xls)
- Testo (.txt, .csv)
- Email (.eml, .msg)
"""

import os
import ctypes
import sys as _sys_init
# FIX: Disabilitare PIR prima di importare PaddlePaddle
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# FIX Python 3.8+: os.add_dll_directory a livello modulo, prima di qualsiasi
# import che possa caricare paddle/CUDA. PATH non viene usato per DLL extension.
import glob as _glob_init
_dll_dirs_module = []
# torch\lib ha le DLL CUDA 12 corrette per Paddle su questo sistema.
# Va in cima alla lista per garantire che cublasLt64_12+cudnn9 vengano trovate PRIMA
# di eventuali DLL incompatibili (cu11 in System32 ecc.).
_torch_lib = next(iter(
    _glob_init.glob(os.path.join(_sys_init.prefix, "Lib", "site-packages", "~orch", "lib"))
    + _glob_init.glob(os.path.join(_sys_init.prefix, "Lib", "site-packages", "torch", "lib"))
), "")
_dll_candidates = (
    ([_torch_lib] if _torch_lib and os.path.isdir(_torch_lib) else [])
    + [
        os.path.join(_sys_init.prefix, "Scripts"),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    ]
    + [p for p in os.environ.get("PATH", "").split(os.pathsep) if p and "cuda" in p.lower()]
)
# Aggiunge anche le cartelle nvidia/*/bin dentro site-packages (pacchetti pip NVIDIA)
for _nvidia_bin in _glob_init.glob(
    os.path.join(_sys_init.prefix, "Lib", "site-packages", "nvidia", "*", "bin")
):
    _dll_candidates.append(_nvidia_bin)
for _dp in _dll_candidates:
    if _dp and os.path.isdir(_dp):
        try:
            _dll_dirs_module.append(os.add_dll_directory(_dp))
        except Exception:
            pass
# Aggiorna anche PATH per Paddle (usa LoadLibrary che rispetta PATH, non add_dll_directory)
_cur_path = os.environ.get("PATH", "")
_extra_paths = os.pathsep.join(d for d in _dll_candidates if d and os.path.isdir(d) and d not in _cur_path)
if _extra_paths:
    os.environ["PATH"] = _extra_paths + os.pathsep + _cur_path

# Pre-carica le DLL CUDA critiche con ctypes per evitare che PaddlePaddle fallisca
# nel trovarle durante l'inferenza (LoadLibrary interno non usa os.add_dll_directory).
import ctypes as _ctypes
_preload_dlls = [
    # CUDA 12 (PyTorch/venv\Scripts) - versione usata da Paddle su questo sistema
    "cudart64_12.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",   # cuDNN 9 da torch/lib - compatible con Paddle cu118
    # CUDA 11 (fallback)
    "cudnn64_8.dll",
    "cublas64_11.dll",
    "cublasLt64_11.dll",
    "cudart64_110.dll",
]
for _dll_name in _preload_dlls:
    for _search_dir in _dll_candidates:
        if not _search_dir or not os.path.isdir(_search_dir):
            continue
        _dll_path = os.path.join(_search_dir, _dll_name)
        if os.path.isfile(_dll_path):
            try:
                _ctypes.WinDLL(_dll_path)
            except Exception:
                pass
            break  # trovata e caricata, prossima DLL

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None  # type: ignore[assignment]
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from loguru import logger

from zt_legal_analysis_core.core.config import settings

# Flag module-level: una volta che la GPU PaddleOCR crasha, tutte le istanze
# successive partono direttamente in CPU-mode senza ritentare la GPU.
_PADDLE_GPU_GLOBALLY_FAILED: bool = False


def _is_paddle_gpu_runtime_available() -> bool:
    """Verifica che le librerie richieste da paddlepaddle-gpu siano disponibili.
    Supporta Windows (DLL check) e Linux (so + cuda device check).
    Su GPU Ada Lovelace / sm_89 (L40S, RTX 4090) con paddle < 3.0 restituisce False
    per evitare SIGSEGV durante ZeroCopyRun()."""
    # --- Linux / macOS ---
    if os.name != "nt":
        import subprocess
        # Se CUDA_VISIBLE_DEVICES è vuoto/none → nessuna GPU visibile
        cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_vis in ("", "-1", "NoDevFiles"):
            return False
        # Verifica presenza della libreria .so paddlepaddle-gpu
        try:
            import paddle  # type: ignore
            if not getattr(paddle, "is_compiled_with_cuda", lambda: False)():
                return False  # CPU-only wheel installato
            # Controlla compute capability: paddle 2.6.x non supporta sm_89 (Ada Lovelace)
            # In questo caso forziamo CPU per evitare SIGSEGV
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                caps = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
                paddle_version = tuple(int(x) for x in getattr(paddle, "__version__", "0.0.0").split(".")[:2])
                for cap in caps:
                    try:
                        major, minor = int(cap.split(".")[0]), int(cap.split(".")[1])
                        # sm_89 = Ada Lovelace (L40S, RTX 4090): richiede paddle >= 3.0
                        if major == 8 and minor >= 9 and paddle_version < (3, 0):
                            logger.warning(
                                f"⚠️ GPU compute capability {cap} (Ada Lovelace) non supportata da "
                                f"paddlepaddle {getattr(paddle, '__version__', '?')} < 3.0. "
                                "PaddleOCR forzato in CPU mode per evitare SIGSEGV."
                            )
                            return False
                        # sm_90 = Hopper (H100): richiede paddle >= 3.0
                        if major >= 9 and paddle_version < (3, 0):
                            logger.warning(
                                f"⚠️ GPU compute capability {cap} non supportata da "
                                f"paddlepaddle {getattr(paddle, '__version__', '?')} < 3.0. "
                                "PaddleOCR forzato in CPU mode."
                            )
                            return False
                    except (ValueError, IndexError):
                        pass
            except Exception:
                pass  # nvidia-smi non disponibile: lasciamo decidere a paddle
            return True
        except ImportError:
            return False

    # --- Windows ---
    import sys

    # Costruisce lista di cartelle dove cercare le DLL
    search_dirs = [
        os.path.join(sys.prefix, "Scripts"),
        r"C:\Windows\System32",
        r"C:\Windows\SysWOW64",
    ]
    # Cartelle nvidia/*/bin in site-packages (pacchetti pip NVIDIA)
    import glob
    for nvidia_bin in glob.glob(
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin")
    ):
        search_dirs.append(nvidia_bin)
    # Cartelle CUDA eventualmente presenti in PATH
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if "cuda" in p.lower() and os.path.isdir(p):
            search_dirs.append(p)

    def _dll_found(name: str) -> bool:
        try:
            ctypes.WinDLL(name)
            return True
        except Exception:
            pass
        return any(os.path.isfile(os.path.join(d, name)) for d in search_dirs)

    # cuDNN 8 è richiesto da entrambi cu118 e cu120
    if not _dll_found("cudnn64_8.dll"):
        return False
    # Verifica CUDA 11.8 (cublas11 + cudart110)
    has_cu11 = _dll_found("cublas64_11.dll") and _dll_found("cudart64_110.dll")
    # Verifica CUDA 12 (cublas12)
    has_cu12 = _dll_found("cublasLt64_12.dll") or _dll_found("cublas64_12.dll")
    return has_cu11 or has_cu12


class DocumentType(Enum):
    """Tipi di documento supportati"""
    PDF = "pdf"
    IMAGE = "image"
    WORD = "word"
    EXCEL = "excel"
    TEXT = "text"
    EMAIL = "email"
    UNKNOWN = "unknown"


# Mapping estensioni -> tipo documento
EXTENSION_MAP: Dict[str, DocumentType] = {
    # PDF
    ".pdf": DocumentType.PDF,
    # Immagini
    ".jpg": DocumentType.IMAGE,
    ".jpeg": DocumentType.IMAGE,
    ".png": DocumentType.IMAGE,
    ".tiff": DocumentType.IMAGE,
    ".tif": DocumentType.IMAGE,
    ".bmp": DocumentType.IMAGE,
    ".gif": DocumentType.IMAGE,
    # Word
    ".docx": DocumentType.WORD,
    ".doc": DocumentType.WORD,
    ".odt": DocumentType.WORD,
    ".rtf": DocumentType.WORD,
    # Excel
    ".xlsx": DocumentType.EXCEL,
    ".xls": DocumentType.EXCEL,
    ".csv": DocumentType.TEXT,
    # Testo
    ".txt": DocumentType.TEXT,
    # Email
    ".eml": DocumentType.EMAIL,
    ".msg": DocumentType.EMAIL,
}


@dataclass
class OCRResult:
    """Risultato dell'elaborazione OCR"""
    text: str
    pages: int
    confidence: float
    method: str  # 'native' | 'ocr' | 'docx' | 'xlsx' | 'text' | 'email'
    document_type: DocumentType = DocumentType.UNKNOWN
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


@dataclass
class OCRRoutingDecision:
    """Decisione di routing OCR document-aware."""
    profile: str
    is_native_candidate: bool
    use_smart_pipeline: bool
    use_florence: bool
    reason: str


@dataclass  
class PageContent:
    """Contenuto di una singola pagina"""
    page_num: int
    text: str
    has_images: bool
    has_tables: bool


class OCREngine:
    """
    Engine per estrazione testo da documenti multi-formato.
    
    Supporta:
    - PDF nativi (estrazione diretta) e scansioni (OCR)
    - Immagini (JPG, PNG, TIFF, BMP)
    - Word (DOCX, DOC)
    - Excel (XLSX, XLS, CSV)
    - Testo (TXT)
    - Email (EML, MSG)
    """
    
    SUPPORTED_EXTENSIONS = list(EXTENSION_MAP.keys())

    PROFILE_SIGNATURE_SENSITIVE = {"contrattuale", "titolo_esecutivo", "garanzia_reale"}
    PROFILE_TABLE_SENSITIVE = {"tabellare_finanziario"}
    
    def __init__(self):
        self.tesseract_available = self._check_tesseract()
        self.poppler_available = self._check_poppler()
        self._paddle_ocr = None
        self._paddle_missing_logged = False
        self._paddle_gpu_precheck_logged = False
        self._paddle_runtime_failed = False
        self._surya_models = None  # cache: (DetectionPredictor, RecognitionPredictor) — Surya v0.7+ API
        self._ocr_cache: dict = {}  # prefetch cache: Path → OCRResult
        logger.info(f"📄 OCREngine supporta: {', '.join(self.SUPPORTED_EXTENSIONS)}")
        
    def _check_tesseract(self) -> bool:
        """Verifica se Tesseract è installato"""
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
            pytesseract.get_tesseract_version()
            logger.info("✅ Tesseract OCR disponibile")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Tesseract non disponibile: {e}")
            logger.warning("Per OCR scansioni, installa: choco install tesseract")
            return False
    
    def _check_poppler(self) -> bool:
        """Verifica se Poppler è installato (per pdf2image)"""
        poppler_path = Path(settings.poppler_path)
        if poppler_path.exists():
            logger.info("✅ Poppler disponibile")
            return True
        else:
            logger.warning("⚠️ Poppler non disponibile")
            logger.warning("Per OCR scansioni, installa: choco install poppler")
            return False
    
    def unload_gpu_models(self):
        """
        Scarica modelli OCR dalla GPU per liberare VRAM.
        Chiamare dopo batch OCR prima di usare LLM.
        """
        # Mantieni modelli se VRAM usata è sotto soglia (RTX 4060 Ti 16GB)
        try:
            from zt_legal_analysis_core.utils.gpu_monitor import get_gpu_monitor
            from zt_legal_analysis_core.core.config import settings

            gpu = get_gpu_monitor()
            stats = gpu.get_stats() if gpu.available else None
            if stats and stats.memory_used < settings.vram_keep_models_threshold_mb:
                logger.debug(
                    "🧠 VRAM sotto soglia, mantengo modelli OCR "
                    f"({stats.memory_used}MB < {settings.vram_keep_models_threshold_mb}MB)"
                )
                return
        except Exception as e:
            logger.debug(f"VRAM check skipped: {e}")

        try:
            from .smart_pipeline import get_smart_pipeline
            pipeline = get_smart_pipeline()
            pipeline.unload_models()
            logger.info("🗑️ Modelli OCR scaricati dalla VRAM")
        except Exception as e:
            logger.debug(f"Unload models: {e}")
        
        # Anche PaddleOCR e Surya locali
        self._paddle_ocr = None
        self._surya_models = None
        
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
    
    def get_document_type(self, filepath: Path) -> DocumentType:
        """Determina il tipo di documento dall'estensione"""
        ext = filepath.suffix.lower()
        return EXTENSION_MAP.get(ext, DocumentType.UNKNOWN)
    
    def is_supported(self, filepath: Path) -> bool:
        """Verifica se il formato è supportato"""
        return filepath.suffix.lower() in EXTENSION_MAP
    
    def prefetch_ocr(self, filepaths: list) -> None:
        """
        Pre-elabora tutti i PDF di un fascicolo con Surya in un'unica chiamata batch.
        I risultati vengono messi in cache: le successive chiamate a process() / extract_text()
        restituiranno i dati dalla cache senza rieseguire Surya.

        Chiamare PRIMA del loop di elaborazione documenti nel fascicolo_aggregator.
        """
        from pdf2image import convert_from_path

        filepaths = [Path(p) for p in filepaths]

        # Identifica i PDF che necessitano OCR e non sono già in cache
        # IMPORTANTE: i PDF con use_smart_pipeline=True (scansioni complesse, contratti)
        # NON vengono batchati — devono passare per la smart pipeline per produrre
        # testo di qualità per il downstream LLM (Qwen 72B).
        # Vengono batchati SOLO i PDF scansionati semplici (use_smart_pipeline=False).
        # In ogni caso i modelli Surya restano cachati dopo il primo caricamento.
        to_batch = []
        for fp in filepaths:
            key = str(fp.resolve())
            if key in self._ocr_cache:
                continue
            doc_type = self.get_document_type(fp)
            if doc_type != DocumentType.PDF:
                continue
            try:
                native_text, _ = self._extract_native(fp)
            except Exception:
                native_text = ""
            routing = self._decide_pdf_routing(fp, native_text)
            # Batcha solo i PDF scansionati che NON richiedono smart pipeline
            if not routing.is_native_candidate and not routing.use_smart_pipeline:
                to_batch.append(fp)

        if not to_batch:
            return

        surya_models = self._get_surya_models()
        if surya_models is None:
            return

        det_predictor, rec_predictor = surya_models

        # Converti tutti i PDF in immagini
        all_images: list = []
        pages_per_pdf: list = []
        valid_paths: list = []

        for fp in to_batch:
            try:
                imgs = convert_from_path(
                    fp,
                    dpi=settings.ocr_dpi,
                    poppler_path=settings.poppler_path,
                )
            except Exception as e:
                logger.warning(f"prefetch_ocr: conversione fallita [{fp.name}]: {e}")
                imgs = []
            pages_per_pdf.append(len(imgs))
            valid_paths.append(fp)
            all_images.extend(imgs)

        if not all_images:
            return

        logger.info(f"🔄 Surya prefetch: {len(all_images)} pagine da {len(valid_paths)} PDF...")
        try:
            import numpy as np
            surya_results = rec_predictor(
                [np.array(img) for img in all_images],
                det_predictor=det_predictor,
            )
        except Exception as e:
            logger.warning(f"prefetch_ocr: Surya batch fallito ({e})")
            return

        import time
        cursor = 0
        for fp, n_pages in zip(valid_paths, pages_per_pdf):
            page_results = surya_results[cursor: cursor + n_pages]
            cursor += n_pages

            pages_text = []
            for p_idx, pr in enumerate(page_results):
                lines = getattr(pr, "text_lines", None) or []
                page_lines = [ln.text for ln in lines if getattr(ln, "text", "").strip()]
                pages_text.append(f"--- PAGINA {p_idx + 1} ---\n" + "\n".join(page_lines))

            text = "\n\n".join(pages_text)
            key = str(fp.resolve())
            self._ocr_cache[key] = OCRResult(
                text=text,
                pages=n_pages,
                confidence=0.90 if text.strip() else 0.0,
                method="surya_batch",
                processing_time=0.0,
                metadata={
                    "document_type": DocumentType.PDF.value,
                    "source_extension": ".pdf",
                    "ocr_profile": "pdf_batch",
                    "routing_reason": "surya_prefetch",
                    "florence_used": False,
                },
                document_type=DocumentType.PDF,
            )
            logger.info(f"  ✅ prefetch [{fp.name}]: {n_pages} pag, {len(text)} chars")

    def extract_text(self, filepath: str) -> Dict[str, Any]:
        """
        Metodo principale: estrae testo da qualsiasi formato supportato.
        
        Returns:
            Dict con 'text', 'pages', 'confidence', 'method', 'document_type'
        """
        result = self.process(Path(filepath))
        return {
            "text": result.text,
            "pages": result.pages,
            "confidence": result.confidence,
            "method": result.method,
            "document_type": result.document_type.value,
            "metadata": result.metadata,
            "errors": result.errors,
        }
    
    def process(self, filepath: Path) -> OCRResult:
        """
        Processa un documento ed estrae il testo.
        Rileva automaticamente il formato e usa il metodo appropriato.
        """
        import time
        start_time = time.time()

        # Controlla cache prefetch
        key = str(Path(filepath).resolve())
        if key in self._ocr_cache:
            logger.debug(f"📦 cache hit: {Path(filepath).name}")
            return self._ocr_cache[key]

        
        filepath = Path(filepath)
        if not filepath.exists():
            return OCRResult(
                text="",
                pages=0,
                confidence=0,
                method="error",
                errors=[f"File non trovato: {filepath}"]
            )
        
        doc_type = self.get_document_type(filepath)
        
        if doc_type == DocumentType.UNKNOWN:
            return OCRResult(
                text="",
                pages=0,
                confidence=0,
                method="error",
                document_type=doc_type,
                errors=[f"Formato non supportato: {filepath.suffix}. Supportati: {', '.join(self.SUPPORTED_EXTENSIONS)}"]
            )
        
        logger.info(f"📄 Elaborazione {doc_type.value}: {filepath.name}")
        
        # Dispatch al metodo appropriato
        if doc_type == DocumentType.PDF:
            result = self._process_pdf(filepath)
        elif doc_type == DocumentType.IMAGE:
            result = self._process_image(filepath)
        elif doc_type == DocumentType.WORD:
            result = self._process_word(filepath)
        elif doc_type == DocumentType.EXCEL:
            result = self._process_excel(filepath)
        elif doc_type == DocumentType.TEXT:
            result = self._process_text(filepath)
        elif doc_type == DocumentType.EMAIL:
            result = self._process_email(filepath)
        else:
            result = OCRResult(
                text="",
                pages=0,
                confidence=0,
                method="error",
                errors=[f"Handler non implementato per: {doc_type.value}"]
            )
        
        result.document_type = doc_type
        result.processing_time = time.time() - start_time
        result.metadata = result.metadata or {}
        result.metadata.setdefault("document_type", doc_type.value)
        result.metadata.setdefault("source_extension", filepath.suffix.lower())
        result.metadata.setdefault("florence_used", False)
        result.metadata.setdefault("ocr_profile", self._default_profile_for_doc_type(doc_type))
        result.metadata.setdefault("routing_reason", self._default_reason_for_doc_type(doc_type))
        
        return result

    def process_batch(self, filepaths: list) -> list:
        """
        Processa più PDF contemporaneamente sfruttando i modelli Surya già caricati.

        Raccoglie le immagini di tutti i PDF scansionati in un unico batch e invoca
        rec_predictor una sola volta — massimizza il throughput su CPU/GPU.
        I documenti non-PDF (Word, immagini, ecc.) vengono processati singolarmente
        in modo sequenziale tramite process().

        Returns:
            Lista di OCRResult nello stesso ordine di filepaths.
        """
        import time
        from pdf2image import convert_from_path

        filepaths = [Path(p) for p in filepaths]
        results: list = [None] * len(filepaths)

        # Separa i PDF che necessitano OCR dagli altri
        # Per "altri" (Word, immagini, testi nativi) usa process() direttamente
        pdf_indices = []     # indici nel batch originale
        pdf_paths   = []     # Path dei PDF da batchare

        for i, fp in enumerate(filepaths):
            doc_type = self.get_document_type(fp)
            if doc_type == DocumentType.PDF:
                # Controlla se è nativo o scansionato
                try:
                    native_text, _ = self._extract_native(fp)
                except Exception:
                    native_text = ""
                routing = self._decide_pdf_routing(fp, native_text)
                if routing.is_native_candidate and not routing.use_smart_pipeline:
                    # PDF nativo: process() diretto, nessun OCR necessario
                    results[i] = self.process(fp)
                else:
                    pdf_indices.append(i)
                    pdf_paths.append(fp)
            else:
                results[i] = self.process(fp)

        if not pdf_paths:
            return results

        # Pre-carica modelli Surya una volta sola
        surya_models = self._get_surya_models()

        if surya_models is None:
            # Surya non disponibile: fallback sequenziale
            logger.warning("⚠️ Surya non disponibile, fallback a process() sequenziale per i PDF")
            for i, fp in zip(pdf_indices, pdf_paths):
                results[i] = self.process(fp)
            return results

        det_predictor, rec_predictor = surya_models

        # Converti tutti i PDF in immagini e tieni traccia di quante pagine per PDF
        all_images   = []
        pages_per_pdf = []   # quante immagini contribuisce ogni PDF
        start_times  = []

        for fp in pdf_paths:
            t0 = time.time()
            try:
                imgs = convert_from_path(
                    fp,
                    dpi=settings.ocr_dpi,
                    poppler_path=settings.poppler_path,
                )
            except Exception as e:
                logger.warning(f"Conversione PDF fallita [{fp.name}]: {e}")
                imgs = []
            start_times.append(t0)
            pages_per_pdf.append(len(imgs))
            all_images.extend(imgs)

        if not all_images:
            for i, fp in zip(pdf_indices, pdf_paths):
                results[i] = self.process(fp)
            return results

        # Singola chiamata Surya su tutte le pagine di tutti i PDF
        logger.info(f"🔄 Surya batch: {len(all_images)} pagine da {len(pdf_paths)} PDF...")
        try:
            import numpy as np
            np_images = [np.array(img) for img in all_images]
            surya_results = rec_predictor(np_images, det_predictor=det_predictor)
        except Exception as e:
            logger.warning(f"Surya batch fallito ({e}), fallback sequenziale")
            for i, fp in zip(pdf_indices, pdf_paths):
                results[i] = self.process(fp)
            return results

        # Riassembla i risultati per PDF
        cursor = 0
        for idx, (orig_i, fp, n_pages, t0) in enumerate(
            zip(pdf_indices, pdf_paths, pages_per_pdf, start_times)
        ):
            page_results = surya_results[cursor: cursor + n_pages]
            cursor += n_pages

            pages_text = []
            for p_idx, page_res in enumerate(page_results):
                lines = getattr(page_res, "text_lines", None) or []
                page_lines = [ln.text for ln in lines if getattr(ln, "text", "").strip()]
                pages_text.append(f"--- PAGINA {p_idx + 1} ---\n" + "\n".join(page_lines))

            text = "\n\n".join(pages_text)
            proc_time = time.time() - t0

            if text.strip():
                logger.info(f"✅ [{fp.name}] {n_pages} pag, {len(text)} chars ({proc_time:.1f}s)")
            else:
                logger.warning(f"⚠️ [{fp.name}] testo vuoto dopo Surya batch")

            results[orig_i] = OCRResult(
                text=text,
                pages=n_pages,
                confidence=0.90 if text.strip() else 0.0,
                method="surya_batch",
                processing_time=proc_time,
                metadata={
                    "document_type": DocumentType.PDF.value,
                    "source_extension": ".pdf",
                    "ocr_profile": "pdf_batch",
                    "routing_reason": "surya_batch",
                    "florence_used": False,
                },
                document_type=DocumentType.PDF,
            )

        return results

    @staticmethod
    def _default_profile_for_doc_type(doc_type: DocumentType) -> str:
        mapping = {
            DocumentType.WORD: "office_text_native",
            DocumentType.EXCEL: "office_tabular_native",
            DocumentType.TEXT: "plain_text_native",
            DocumentType.EMAIL: "email_native",
            DocumentType.IMAGE: "image_ocr",
            DocumentType.PDF: "pdf",
        }
        return mapping.get(doc_type, "generico")

    @staticmethod
    def _default_reason_for_doc_type(doc_type: DocumentType) -> str:
        mapping = {
            DocumentType.WORD: "Word: parser nativo senza Florence",
            DocumentType.EXCEL: "Excel: parser tabellare nativo senza Florence",
            DocumentType.TEXT: "TXT/CSV: lettura diretta senza OCR",
            DocumentType.EMAIL: "EML/MSG: parser email nativo senza Florence",
            DocumentType.IMAGE: "Immagine: OCR visuale",
            DocumentType.PDF: "PDF: routing dinamico nativo/scansione",
        }
        return mapping.get(doc_type, "routing standard")

    def _infer_pdf_profile(self, pdf_path: Path, sample_text: str = "") -> str:
        """Inferenza tipologia documento da filename + contenuto (routing OCR)."""
        name = (pdf_path.name or "").lower()
        content = (sample_text or "").lower()

        def has_any(keywords: List[str]) -> bool:
            return any(k in name for k in keywords) or any(k in content for k in keywords)

        if has_any(["estratto conto", "estratto_conto", "movimenti", "saldo", "piano ammortamento", "fatture", "contabili", "dare", "avere"]):
            return "tabellare_finanziario"
        if has_any(["fideiuss", "contratto", "mutuo", "finanziamento", "scrittura privata", "clausola", "sottoscritto", "parte contraente"]):
            return "contrattuale"
        if has_any(["decreto ingiuntivo", "decreto_ingiuntivo", "sentenza", "precetto", "pignoramento", "citazione", "tribunale", "giudice"]):
            return "titolo_esecutivo"
        if has_any(["ipoteca", "visura ipotecaria", "perizia", "garanzie", "catastale", "immobile", "foglio", "particella"]):
            return "garanzia_reale"
        if has_any(["corrispondenza", "lettera", "comunicazione", "email", "pec", "oggetto:", "mittente", "destinatario"]):
            return "corrispondenza"
        return "generico"

    def _decide_pdf_routing(self, pdf_path: Path, native_text: str) -> OCRRoutingDecision:
        """Policy di routing OCR alto livello per PDF."""
        profile = self._infer_pdf_profile(pdf_path, native_text)
        native_len = len((native_text or "").strip())
        native_ok = native_len > 200 and self._is_text_quality_ok(native_text or "")

        # Regole uso Florence:
        # - preferibile su documenti signature-sensitive (contratti/titoli/garanzie)
        # - su scansioni quando disponibile
        florence_enabled = bool(getattr(settings, "enable_florence2", False))
        smart_enabled = bool(getattr(settings, "enable_smart_pipeline", True))

        if not smart_enabled:
            return OCRRoutingDecision(
                profile=profile,
                is_native_candidate=native_ok,
                use_smart_pipeline=False,
                use_florence=False,
                reason="smart pipeline disabilitata da configurazione",
            )

        if not native_ok:
            return OCRRoutingDecision(
                profile=profile,
                is_native_candidate=False,
                use_smart_pipeline=True,
                use_florence=florence_enabled,
                reason="pdf non nativo/scansionato: serve OCR con routing",
            )

        if profile in self.PROFILE_SIGNATURE_SENSITIVE:
            return OCRRoutingDecision(
                profile=profile,
                is_native_candidate=True,
                use_smart_pipeline=True,
                use_florence=florence_enabled,
                reason="documento signature-sensitive: validazione firma/layout prioritaria",
            )

        return OCRRoutingDecision(
            profile=profile,
            is_native_candidate=True,
            use_smart_pipeline=False,
            use_florence=False,
            reason="pdf nativo non signature-sensitive: estrazione nativa prioritaria",
        )
    
    def _process_pdf(self, pdf_path: Path) -> OCRResult:
        """
        Processa un PDF ed estrae il testo.
        
        Strategia (Gennaio 2026):
        1. Smart Pipeline (Florence-2 + routing intelligente) - NUOVO
        2. PyMuPDF4LLM (ottimizzato per LLM)
        3. PyMuPDF (standard)
        4. OCR (scansioni con preprocessing)
        """
        import time
        start_time = time.time()
        
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return OCRResult(
                text="",
                pages=0,
                confidence=0,
                method="error",
                errors=[f"File non trovato: {pdf_path}"]
            )
        
        # Pre-check: PDF corrotto o vuoto?
        if pdf_path.stat().st_size < 100:
            logger.warning(f"⚠️ PDF troppo piccolo (< 100 bytes): {pdf_path.name}")
            return OCRResult(
                text="",
                pages=0,
                confidence=0,
                method="error",
                errors=["PDF vuoto o corrotto (dimensione < 100 bytes)"]
            )
        
        # 0. Routing PDF document-aware (tipo documento + nativo/scansionato)
        native_probe_text = ""
        native_probe_pages = 0
        try:
            native_probe_text, native_probe_pages = self._extract_native(pdf_path)
        except Exception:
            native_probe_text, native_probe_pages = "", 0

        routing = self._decide_pdf_routing(pdf_path, native_probe_text)
        logger.info(
            f"🧭 OCR routing [{pdf_path.name}] profile={routing.profile}, "
            f"native={routing.is_native_candidate}, smart={routing.use_smart_pipeline}, "
            f"florence={routing.use_florence}"
        )

        if routing.use_smart_pipeline:
            try:
                from .smart_pipeline import get_smart_pipeline

                pipeline = get_smart_pipeline()

                result = pipeline.process(pdf_path, use_florence=routing.use_florence)

                if result.text and len(result.text.strip()) > 100:
                    processing_time = time.time() - start_time
                    logger.info(
                        f"📄 PDF smart pipeline: {pdf_path.name} "
                        f"({result.pages} pag, {result.strategy.value}, "
                        f"firma={'✓' if result.has_signature else '✗'})"
                    )

                    metadata = result.metadata.copy()
                    metadata["has_signature"] = result.has_signature
                    metadata["signature_count"] = result.signature_count
                    metadata["has_stamp"] = result.has_stamp
                    metadata["is_scanned"] = result.is_scanned
                    metadata["quality_score"] = result.quality_score
                    metadata["warnings"] = result.warnings
                    metadata["ocr_profile"] = routing.profile
                    metadata["routing_reason"] = routing.reason
                    metadata["florence_used"] = bool(routing.use_florence)

                    return OCRResult(
                        text=result.text,
                        pages=result.pages,
                        confidence=result.confidence,
                        method=f"smart_pipeline_{result.strategy.value}",
                        processing_time=processing_time,
                        metadata=metadata
                    )
                else:
                    logger.debug("Smart pipeline fallback: risultato insufficiente")
            except ImportError as e:
                logger.debug(f"Smart pipeline non disponibile (dipendenze mancanti): {e}")
            except Exception as e:
                logger.warning(f"Smart pipeline fallita: {e}")
        
        # 1. Prova ENHANCED PARSER (PyMuPDF4LLM + fallback)
        try:
            from .enhanced_parser import get_parser
            
            parser = get_parser()
            parsed = parser.parse(str(pdf_path))
            
            if parsed.success and parsed.text and len(parsed.text.strip()) > 100:
                processing_time = time.time() - start_time
                logger.info(f"📄 PDF parsed (enhanced): {pdf_path.name} ({parsed.metadata.get('pages', 0)} pagine)")
                return OCRResult(
                    text=parsed.text,
                    pages=parsed.metadata.get('pages', 0),
                    confidence=0.95,  # PyMuPDF4LLM è ottimizzato per LLM
                    method="enhanced_parser",
                    processing_time=processing_time,
                    metadata=parsed.metadata
                )
            else:
                logger.debug(f"Enhanced parser fallback: {parsed.error}")
        except Exception as e:
            logger.debug(f"Enhanced parser non disponibile: {str(e)}")
        
        # 2. Fallback a estrazione nativa PyMuPDF standard
        try:
            native_text = native_probe_text
            page_count = native_probe_pages
            if page_count <= 0:
                native_text, page_count = self._extract_native(pdf_path)
        except Exception as e:
            logger.error(f"❌ PDF corrotto non processabile: {pdf_path.name} - {e}")
            return OCRResult(
                text="",
                pages=0,
                confidence=0,
                method="error",
                errors=[f"PDF corrotto: {str(e)}"]
            )
        
        # Se ha abbastanza testo e qualità accettabile, usa quello
        if native_text and self._is_text_quality_ok(native_text):
            native_pages = self._split_pages(native_text)

            # Se mancano pagine testuali, prova OCR per riempire le pagine assenti
            if page_count and len(native_pages) < page_count:
                logger.info(f"📄 PDF nativo incompleto ({len(native_pages)}/{page_count} pagine) - integrazione OCR")
                ocr_text = ""
                confidence = 0.90
                if self.tesseract_available and self.poppler_available:
                    ocr_text, confidence = self._perform_ocr(pdf_path)
                else:
                    ocr_text = self._perform_ocr_paddle_pdf(pdf_path)
                    confidence = 0.85 if ocr_text else 0.60

                if ocr_text:
                    ocr_pages = self._split_pages(ocr_text)
                    merged_pages = []
                    for i in range(1, page_count + 1):
                        page_text = native_pages.get(i) or ocr_pages.get(i, "")
                        merged_pages.append(f"--- PAGINA {i} ---\n{page_text}")
                    processing_time = time.time() - start_time
                    return OCRResult(
                        text="\n\n".join(merged_pages),
                        pages=page_count,
                        confidence=min(confidence, 0.90),
                        method="native+ocr",
                        processing_time=processing_time
                    )

            processing_time = time.time() - start_time
            logger.info(f"📄 PDF nativo (fallback): {pdf_path.name} ({page_count} pagine)")
            return OCRResult(
                text=native_text,
                pages=page_count,
                confidence=0.90,
                method="native_fallback",
                processing_time=processing_time
                ,
                metadata={
                    "ocr_profile": routing.profile,
                    "routing_reason": routing.reason,
                    "florence_used": False,
                    "is_scanned": False,
                }
            )
        
        # 2. Fallback a OCR
        if self.tesseract_available and self.poppler_available:
            logger.info(f"🔍 Avvio OCR per: {pdf_path.name}")
            ocr_text, confidence = self._perform_ocr(pdf_path)
            processing_time = time.time() - start_time
            return OCRResult(
                text=ocr_text,
                pages=page_count,
                confidence=confidence,
                method="ocr",
                processing_time=processing_time,
                metadata={
                    "ocr_profile": routing.profile,
                    "routing_reason": routing.reason,
                    "florence_used": False,
                    "is_scanned": True,
                }
            )
        # 2b. OCR con PaddleOCR via PyMuPDF (senza Poppler/Tesseract)
        paddle_text = self._perform_ocr_paddle_pdf(pdf_path)
        if paddle_text:
            processing_time = time.time() - start_time
            return OCRResult(
                text=paddle_text,
                pages=page_count,
                confidence=0.85,
                method="paddle_ocr",
                processing_time=processing_time,
                metadata={
                    "ocr_profile": routing.profile,
                    "routing_reason": routing.reason,
                    "florence_used": False,
                    "is_scanned": True,
                }
            )
        else:
            processing_time = time.time() - start_time
            return OCRResult(
                text=native_text or "",
                pages=page_count,
                confidence=0.5,
                method="native_fallback",
                processing_time=processing_time,
                metadata={
                    "ocr_profile": routing.profile,
                    "routing_reason": routing.reason,
                    "florence_used": False,
                },
                errors=["OCR non disponibile, risultato potrebbe essere incompleto"]
            )

    def _is_text_quality_ok(self, text: str) -> bool:
        """Valuta se il testo nativo è di qualità sufficiente; altrimenti forza OCR."""
        cleaned = text.strip()
        if len(cleaned) < 200:
            return False
        letters = sum(1 for c in cleaned if c.isalpha())
        ratio = letters / max(len(cleaned), 1)
        return ratio >= 0.2

    def _split_pages(self, text: str) -> Dict[int, str]:
        """Divide un testo con marker '--- PAGINA X ---' in mappa pagina->testo."""
        pages: Dict[int, str] = {}
        if not text:
            return pages
        pattern = re.compile(r"---\s*PAGINA\s*(\d+)\s*---", re.IGNORECASE)
        matches = list(pattern.finditer(text))
        if not matches:
            return pages

        for idx, match in enumerate(matches):
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            try:
                page_num = int(match.group(1))
            except Exception:
                continue
            pages[page_num] = text[start:end].strip()

        return pages
    
    def _extract_native(self, pdf_path: Path) -> Tuple[str, int]:
        """Estrae testo da PDF nativo usando PyMuPDF con gestione errori robusta"""
        try:
            # Apri con gestione errori e timeout
            import warnings
            import io
            import sys
            
            # Sopprimi warning MuPDF che causano blocchi
            warnings.filterwarnings('ignore', category=UserWarning)
            
            # Cattura stderr per evitare blocchi su errori MuPDF
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
            try:
                doc = fitz.open(pdf_path)
            except Exception as open_error:
                sys.stderr = old_stderr
                logger.warning(f"⚠️ Errore apertura PDF {pdf_path.name}: {open_error}")
                return "", 0
            finally:
                # Ripristina stderr
                if sys.stderr != old_stderr:
                    sys.stderr = old_stderr
            
            pages = []
            page_count = 0
            
            # Itera pagine con try/except per ogni pagina
            try:
                page_count = len(doc)
            except Exception as e:
                logger.warning(f"⚠️ Impossibile contare pagine PDF {pdf_path.name}: {e}")
                page_count = 0
            
            # Se page_count è 0 o negativo, il PDF è corrotto
            if page_count <= 0:
                logger.error(f"❌ PDF corrotto o vuoto: {pdf_path.name}")
                try:
                    doc.close()
                except:
                    pass
                return "", 0
            
            for page_num in range(page_count):
                try:
                    page = doc.load_page(page_num)
                    text = page.get_text("text")
                    if text and text.strip():
                        pages.append(f"--- PAGINA {page_num + 1} ---\n{text}")
                except Exception as page_error:
                    logger.debug(f"Pagina {page_num + 1} non leggibile: {page_error}")
                    continue
            
            try:
                doc.close()
            except:
                pass
            
            return "\n\n".join(pages), page_count
            
        except Exception as e:
            logger.error(f"Errore estrazione nativa: {e}")
            return "", 0
    
    def _get_surya_models(self):
        """Carica e cachea i modelli Surya (API v0.7+) sul device configurato."""
        if self._surya_models is not None:
            return self._surya_models if self._surya_models is not False else None
        # ── Singleton bridge: riutilizza modelli già caricati da SmartDocumentPipeline
        # (evita il secondo caricamento quando OCREngine è usato come fallback)
        try:
            from .smart_pipeline import SmartDocumentPipeline as _SDP
            _shared = _SDP._class_surya_models
            if _shared is not None and _shared is not False:
                self._surya_models = _shared
                logger.debug("♻️ Surya models riutilizzati da SmartDocumentPipeline (no doppio caricamento)")
                return self._surya_models
        except Exception:
            # ImportError: dipendenza mancante
            # AttributeError: modulo parzialmente importato (circular import in-flight)
            # Qualsiasi altra eccezione: prosegui con il caricamento autonomo
            pass
        # Forza offline PRIMA di qualsiasi import Surya — evita timeout su models.datalab.to
        import os as _os
        _os.environ["HF_HUB_OFFLINE"] = "1"
        _os.environ["TRANSFORMERS_OFFLINE"] = "1"
        rec_path = _os.environ.get("RECOGNITION_MODEL_CHECKPOINT", "")
        det_path = _os.environ.get("DETECTOR_MODEL_CHECKPOINT", "")
        if not rec_path or not det_path:
            logger.warning("⚠️ Surya disabilitato: RECOGNITION_MODEL_CHECKPOINT / DETECTOR_MODEL_CHECKPOINT non impostate")
            self._surya_models = False
            return None
        try:
            from surya.recognition import RecognitionPredictor  # type: ignore
            from surya.detection import DetectionPredictor      # type: ignore
            from surya.foundation import FoundationPredictor    # type: ignore
        except (ImportError, OSError) as _ie:
            logger.warning(f"Surya: import fallito — {type(_ie).__name__}: {_ie}")
            self._surya_models = False
            return None
        try:
            # Patch SuryaDecoderConfig per Surya v0.17.1 (pad_token_id aggiunto in v0.18+)
            try:
                import json as _json
                from surya.common.surya import config as _surya_cfg
                if not hasattr(_surya_cfg.SuryaDecoderConfig, 'pad_token_id'):
                    _rec_cfg_path = _os.path.join(rec_path, 'config.json')
                    with open(_rec_cfg_path) as _f:
                        _rc = _json.load(_f)
                    _dec = _rc.get('decoder', _rc)
                    _pad = _dec.get('pad_token_id') or _dec.get('eos_token_id') or _rc.get('eos_token_id')
                    _surya_cfg.SuryaDecoderConfig.pad_token_id = _pad
                    logger.info(f"✅ Patched SuryaDecoderConfig.pad_token_id={_pad}")
            except Exception as _pe:
                logger.warning(f"Surya config patch fallita: {_pe}")
            # Patch ROPE_INIT_FUNCTIONS per 'default' rope_type (aggiunto in transformers>=4.46)
            try:
                import surya.common.surya.decoder as _dec_mod
                import torch as _torch
                _rope_fns = getattr(_dec_mod, 'ROPE_INIT_FUNCTIONS', None)
                if _rope_fns is None:
                    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _rope_fns
                if 'default' not in _rope_fns:
                    def _rope_default(config, device=None):
                        _base = getattr(config, 'rope_theta', 10000.0)
                        _hdim = getattr(config, 'head_dim', None) or (config.hidden_size // config.num_attention_heads)
                        _inv = 1.0 / (_base ** (_torch.arange(0, _hdim, 2, dtype=_torch.float32) / _hdim))
                        return _inv, 1.0
                    _rope_fns['default'] = _rope_default
                    logger.info("✅ Patched ROPE_INIT_FUNCTIONS['default']")
            except Exception as _re:
                logger.warning(f"Surya ROPE patch fallita: {_re}")
            # Patch 3: all_tied_weights_keys — transformers >= 4.46 lo chiama durante from_pretrained
            try:
                from surya.common.pretrained import SuryaPreTrainedModel as _SuryaPTM
                def _atk_get(self):
                    # transformers >= 4.46 calls .keys() on this — must be a dict
                    all_t = list(getattr(self.__class__, '_tied_weights_keys', None) or [])
                    return {k: k for k in all_t}
                _SuryaPTM.all_tied_weights_keys = property(_atk_get, lambda self, v: None)
                logger.info("✅ Patched SuryaPreTrainedModel.all_tied_weights_keys")
            except Exception as _pe3:
                logger.warning(f"Patch all_tied_weights_keys fallita: {_pe3}")
            # Patch 3b: tie_weights() — transformers >= 4.46 chiama tie_weights(missing_keys=..., recompute_mapping=False)
            try:
                from surya.common.surya import SuryaModel as _SuryaModel
                _orig_tie = _SuryaModel.tie_weights
                def _tie_weights_compat(self, **kwargs):
                    _orig_tie(self)
                _SuryaModel.tie_weights = _tie_weights_compat
                logger.info("✅ Patched SuryaModel.tie_weights")
            except Exception as _pe3b:
                logger.warning(f"Patch tie_weights fallita: {_pe3b}")
            # Patch 3c: _tie_or_clone_weights — rimossa in transformers >= 4.46, implementazione diretta
            try:
                from surya.common.pretrained import SuryaPreTrainedModel as _SuryaPTM
                import torch.nn.functional as _F_tie
                def _tie_or_clone_weights_impl(self, output_embeddings, input_embeddings):
                    output_embeddings.weight = input_embeddings.weight
                    if getattr(output_embeddings, "bias", None) is not None:
                        output_embeddings.bias.data = _F_tie.pad(
                            output_embeddings.bias.data,
                            (0, output_embeddings.weight.shape[0] - output_embeddings.bias.shape[0]),
                            "constant", 0,
                        )
                    if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
                        output_embeddings.out_features = input_embeddings.num_embeddings
                _SuryaPTM._tie_or_clone_weights = _tie_or_clone_weights_impl
                logger.info("✅ Patched SuryaPreTrainedModel._tie_or_clone_weights")
            except Exception as _pe3c:
                logger.warning(f"Patch _tie_or_clone_weights fallita: {_pe3c}")
            # Patch 4: forza defaults SuryaDecoderConfig/SuryaModelConfig dal config.json del checkpoint
            # Se manca config.json usa valori hardcoded del checkpoint 2025_09_23 (hidden=1280, 12 layers)
            try:
                import json as _json_cfg
                from surya.common.surya.decoder.config import SuryaDecoderConfig as _SDecCfg
                from surya.common.surya.config import SuryaModelConfig as _SModCfg
                _cfg_file = _os.path.join(rec_path, 'config.json')
                if _os.path.exists(_cfg_file):
                    with open(_cfg_file) as _fcfg:
                        _cfg_raw = _json_cfg.load(_fcfg)
                    _dec_raw = _cfg_raw.get('decoder', {})
                else:
                    # Fallback: valori noti del checkpoint text_recognition/2025_09_23
                    _cfg_raw = {'vocab_size': 67577}
                    _dec_raw = {}
                _ckpt_hs  = _dec_raw.get('hidden_size', 1280)
                _ckpt_nl  = _dec_raw.get('num_hidden_layers', 12)
                _ckpt_is  = _dec_raw.get('intermediate_size', 6400)
                _ckpt_na  = _dec_raw.get('num_attention_heads', 16)
                _ckpt_nkv = _dec_raw.get('num_key_value_heads', 4)
                _ckpt_dv  = _dec_raw.get('vocab_size', 218225)
                _ckpt_tv  = _cfg_raw.get('vocab_size', 67577)
                _OrigDecInit = _SDecCfg.__init__
                def _patched_dec_init(self_,
                                     vocab_size=_ckpt_dv, hidden_size=_ckpt_hs,
                                     intermediate_size=_ckpt_is, num_hidden_layers=_ckpt_nl,
                                     num_attention_heads=_ckpt_na, num_key_value_heads=_ckpt_nkv,
                                     **kw):
                    _OrigDecInit(self_, vocab_size=vocab_size, hidden_size=hidden_size,
                                 intermediate_size=intermediate_size, num_hidden_layers=num_hidden_layers,
                                 num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads,
                                 **kw)
                _SDecCfg.__init__ = _patched_dec_init
                _OrigModInit = _SModCfg.__init__
                def _patched_mod_init(self_, vocab_size=_ckpt_tv, **kw_):
                    _OrigModInit(self_, vocab_size=vocab_size, **kw_)
                _SModCfg.__init__ = _patched_mod_init
                logger.info(f"✅ Patched SuryaDecoderConfig defaults: hidden={_ckpt_hs}, layers={_ckpt_nl}, dec_vocab={_ckpt_dv}, mod_vocab={_ckpt_tv}")
            except Exception as _pe4:
                logger.warning(f"Patch SuryaDecoderConfig/ModelConfig defaults fallita: {_pe4}")
            device = _os.environ.get("TORCH_DEVICE") or getattr(settings, "surya_device", "cpu")
            logger.info(f"🔄 Carico Surya OCR su {device} (locale: {det_path})...")
            _os.environ.setdefault("TORCH_DEVICE", device)
            det_predictor = DetectionPredictor(checkpoint=det_path, device=device)
            fp = FoundationPredictor(checkpoint=rec_path, device=device, attention_implementation="eager")
            rec_predictor = RecognitionPredictor(foundation_predictor=fp)
            self._surya_models = (det_predictor, rec_predictor)
            logger.info(f"✅ Surya OCR pronto su {device}")
            return self._surya_models
        except Exception as e:
            import traceback as _tb
            logger.warning(f"Surya: caricamento modelli fallito ({e}):\n{_tb.format_exc()}")
            self._surya_models = False
            return None

    def _perform_ocr_surya(self, images: list) -> str:
        """
        OCR con Surya v0.7+ (transformer-based) — compatibile con qualsiasi GPU/CPU.
        Usa modelli cachati; restituisce stringa vuota se Surya non è disponibile.
        """
        models = self._get_surya_models()
        if models is None:
            return ""
        try:
            det_predictor, rec_predictor = models
            results = rec_predictor(images, det_predictor=det_predictor)
            pages_text = []
            for page_idx, page_result in enumerate(results):
                lines = getattr(page_result, "text_lines", None) or []
                page_lines = [ln.text for ln in lines if getattr(ln, "text", "").strip()]
                pages_text.append(f"--- PAGINA {page_idx + 1} ---\n" + "\n".join(page_lines))
            text = "\n\n".join(pages_text)
            if text.strip():
                logger.info(f"✅ Surya OCR: {len(text)} chars su {len(images)} pagine")
            return text
        except Exception as e:
            logger.warning(f"Surya OCR fallito: {e}")
            return ""

    def _perform_ocr(self, pdf_path: Path) -> Tuple[str, float]:
        """Esegue OCR usando Tesseract"""
        try:
            import pytesseract
            from pdf2image import convert_from_path
            
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
            
            # Converti PDF in immagini
            images = convert_from_path(
                pdf_path,
                dpi=settings.ocr_dpi,
                poppler_path=settings.poppler_path
            )

            # Prova Surya OCR se disponibile (compatibile con L40S sm_89, nessun crash cuDNN)
            surya_text = self._perform_ocr_surya(images)
            if surya_text:
                return surya_text, 0.90

            # Prova PaddleOCR se abilitato (con batch processing)
            paddle_text = self._perform_ocr_paddle(images, batch_size=4)
            if paddle_text:
                return paddle_text, 0.85

            # Tesseract OCR con parallelizzazione pagine
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def process_tesseract_page(args):
                i, image = args
                try:
                    page_text = pytesseract.image_to_string(
                        image,
                        lang=settings.ocr_language,
                        config='--oem 3 --psm 6'
                    )
                    # Confidence
                    conf = 80
                    try:
                        data = pytesseract.image_to_data(
                            image,
                            lang=settings.ocr_language,
                            output_type=pytesseract.Output.DICT
                        )
                        confidences = [int(c) for c in data['conf'] if int(c) > 0]
                        if confidences:
                            conf = sum(confidences) / len(confidences)
                    except:
                        pass
                    return i, page_text, conf
                except Exception as e:
                    logger.warning(f"Tesseract pagina {i+1} fallito: {e}")
                    return i, "", 0
            
            all_text = {}
            all_confidence = {}
            
            # Batch parallelo (max 4 thread per evitare saturazione CPU)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(process_tesseract_page, (i, img)): i
                    for i, img in enumerate(images)
                }
                for future in as_completed(futures):
                    try:
                        idx, text, conf = future.result(timeout=60)
                        all_text[idx] = text
                        all_confidence[idx] = conf
                    except Exception:
                        pass
            
            # Riordina per indice pagina
            final_text = []
            total_confidence = 0
            for i in range(len(images)):
                page_text = all_text.get(i, "")
                final_text.append(f"--- PAGINA {i + 1} ---\n{page_text}")
                total_confidence += all_confidence.get(i, 80)

            avg_confidence = (total_confidence / len(images)) / 100 if images else 0
            return "\n\n".join(final_text), avg_confidence
            
        except Exception as e:
            logger.error(f"Errore OCR: {e}")
            return "", 0.0

    def _perform_ocr_paddle_pdf(self, pdf_path: Path) -> str:
        """Esegue OCR su PDF usando PaddleOCR e rendering PyMuPDF."""
        try:
            ocr = self._get_paddle_ocr()
            if not ocr:
                return ""

            import fitz
            from PIL import Image

            doc = fitz.open(pdf_path)
            max_pages = getattr(settings, "ocr_max_pages", 10)
            pages_to_process = doc.page_count if not max_pages or max_pages <= 0 else min(doc.page_count, max_pages)

            images = []
            for page_index in range(pages_to_process):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                mode = "RGB" if pix.alpha == 0 else "RGBA"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                if mode == "RGBA":
                    img = img.convert("RGB")
                images.append(img)

            doc.close()

            paddle_text = self._perform_ocr_paddle(images)
            return paddle_text or ""
        except Exception as e:
            logger.warning(f"PaddleOCR PDF fallito: {e}")
            return ""
    
    def _process_image(self, image_path: Path) -> OCRResult:
        """
        Processa un'immagine con OCR.
        Supporta: JPG, JPEG, PNG, TIFF, BMP, GIF
        
        Usa PyMuPDF come fallback se Tesseract non disponibile.
        """
        # Prova prima con PyMuPDF (più veloce, nessuna dipendenza esterna)
        try:
            import fitz
            from PIL import Image
            
            # Converti immagine in PDF temporaneo per PyMuPDF
            pil_image = Image.open(image_path)
            
            # PyMuPDF può aprire immagini direttamente
            doc = fitz.open(image_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Prima prova a estrarre testo embedded (se esiste)
                page_text = page.get_text()
                
                if page_text.strip():
                    text_parts.append(page_text)
                else:
                    # Se non c'è testo embedded, usa Surya → PaddleOCR → Tesseract
                    surya_text = self._perform_ocr_surya([pil_image])
                    if surya_text:
                        text_parts.append(surya_text)
                        continue
                    paddle_text = self._perform_ocr_paddle([pil_image])
                    if paddle_text:
                        text_parts.append(paddle_text)
                    elif self.tesseract_available:
                        try:
                            import pytesseract
                            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
                            
                            # Estrai immagine dalla pagina
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
                            img_data = pix.tobytes("png")
                            from io import BytesIO
                            img = Image.open(BytesIO(img_data))
                            
                            ocr_text = pytesseract.image_to_string(
                                img,
                                lang=settings.ocr_language,
                                config='--oem 3 --psm 6'
                            )
                            text_parts.append(ocr_text)
                        except Exception as e:
                            logger.warning(f"OCR Tesseract fallito per {image_path.name}: {e}")
                            text_parts.append(f"[Immagine non processabile: {image_path.name}]")
                    else:
                        text_parts.append(f"[Immagine scansionata: {image_path.name} - OCR non disponibile]")
            
            doc.close()
            text = "\n\n".join(text_parts)
            confidence = 0.7 if text.strip() else 0.0
            
            logger.info(f"🖼️ Immagine processata con PyMuPDF: {image_path.name} ({len(text)} chars)")
            
            return OCRResult(
                text=text,
                pages=1,
                confidence=confidence,
                method="ocr_image",
                metadata={"size": pil_image.size, "mode": pil_image.mode}
            )
            
        except Exception as e:
            logger.error(f"Errore OCR immagine: {e}")
            return OCRResult(
                text="",
                pages=1,
                confidence=0,
                method="error",
                errors=[str(e)]
            )

    def _get_paddle_ocr(self):
        """Inizializza PaddleOCR se disponibile e abilitato."""
        global _PADDLE_GPU_GLOBALLY_FAILED
        if not getattr(settings, "enable_paddleocr", False):
            return None

        if self._paddle_runtime_failed:
            return None

        if self._paddle_ocr is not None:
            return self._paddle_ocr

        # Se un'altra istanza ha già fallito GPU → parto subito in CPU
        if _PADDLE_GPU_GLOBALLY_FAILED:
            try:
                setattr(settings, "paddleocr_use_gpu", False)
            except Exception:
                pass

        try:
            # Fix per errori PaddleOCR/PaddleX
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            # Fix protobuf incompatibility con paddlepaddle 2.6.x
            os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
            os.environ.setdefault("FLAGS_enable_pir_api", "0")
            os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
            os.environ.setdefault("FLAGS_use_mkldnn", "0")

            # Python 3.8+: DLL search via PATH non funziona per estensioni C++.
            # Usa la stessa lista _dll_candidates definita a livello modulo
            # (torch/lib in testa = cu12 DLLs corretti per questo sistema).
            import sys as _sys
            import glob as _glob
            _dll_dirs = []
            # torch/lib prima di tutto
            _torch_lib_local = next(iter(
                _glob.glob(os.path.join(_sys.prefix, "Lib", "site-packages", "~orch", "lib"))
                + _glob.glob(os.path.join(_sys.prefix, "Lib", "site-packages", "torch", "lib"))
            ), "")
            _dll_search_paths = (
                ([_torch_lib_local] if _torch_lib_local and os.path.isdir(_torch_lib_local) else [])
                + [os.path.join(_sys.prefix, "Scripts")]
                + list(_glob.glob(os.path.join(_sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin")))
                + [
                    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
                    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
                ]
            )
            for _p in os.environ.get("PATH", "").split(os.pathsep):
                if _p and "cuda" in _p.lower() and os.path.isdir(_p):
                    _dll_search_paths.append(_p)
            # os.add_dll_directory per estensioni Python
            for _dpath in _dll_search_paths:
                try:
                    _dll_dirs.append(os.add_dll_directory(_dpath))
                except Exception:
                    pass
            # Aggiunge anche a PATH (usato da PaddlePaddle internamente via LoadLibrary)
            _current_path = os.environ.get("PATH", "")
            _extra = os.pathsep.join(
                d for d in _dll_search_paths if d not in _current_path
            )
            if _extra:
                os.environ["PATH"] = _extra + os.pathsep + _current_path

            from paddleocr import PaddleOCR

            use_gpu_requested = bool(getattr(settings, "paddleocr_use_gpu", True))
            if use_gpu_requested and not _is_paddle_gpu_runtime_available():
                if not self._paddle_gpu_precheck_logged:
                    logger.warning("⚠️ DLL CUDA richieste da PaddleOCR non trovate (cudnn64_8 + cublas64_11 o cublas64_12). PaddleOCR avviato in CPU.")
                    self._paddle_gpu_precheck_logged = True
                use_gpu_requested = False
                try:
                    setattr(settings, "paddleocr_use_gpu", False)
                except Exception:
                    pass

            kwargs = {
                "use_angle_cls": True,
                "lang": getattr(settings, "paddleocr_lang", "it"),
                "show_log": False,
                "enable_mkldnn": False,   # previene crash oneDNN/MKL-DNN su Windows
            }

            try:
                import inspect

                params = inspect.signature(PaddleOCR.__init__).parameters
                if "use_gpu" in params:
                    kwargs["use_gpu"] = use_gpu_requested
            except Exception:
                pass

            def _build_ocr(local_kwargs: Dict[str, Any]):
                try:
                    return PaddleOCR(**local_kwargs)
                except TypeError:
                    local_kwargs = dict(local_kwargs)
                    local_kwargs.pop("use_gpu", None)
                    return PaddleOCR(**local_kwargs)

            if use_gpu_requested:
                try:
                    self._paddle_ocr = _build_ocr(kwargs)
                except Exception as gpu_exc:
                    msg = str(gpu_exc).lower()
                    gpu_dynload_error = (
                        "cudnn64_8.dll" in msg
                        or "cublaslt64_12" in msg
                        or "cublas64_12" in msg
                        or "could not locate" in msg
                        or "error code is 126" in msg
                        or "preconditionnotmet" in msg
                        or "cudnn" in msg
                    )
                    if not gpu_dynload_error:
                        raise

                    logger.warning("⚠️ PaddleOCR GPU non disponibile (cuDNN/CUDA). Fallback automatico a CPU.")
                    try:
                        setattr(settings, "paddleocr_use_gpu", False)
                    except Exception:
                        pass
                    cpu_kwargs = dict(kwargs)
                    cpu_kwargs["use_gpu"] = False
                    self._paddle_ocr = _build_ocr(cpu_kwargs)
            else:
                self._paddle_ocr = _build_ocr(kwargs)

            logger.info("✅ PaddleOCR inizializzato")
            return self._paddle_ocr
        except Exception as e:
            if not self._paddle_missing_logged:
                logger.warning(f"⚠️ PaddleOCR non disponibile: {e}")
                self._paddle_missing_logged = True
            self._paddle_ocr = None
            return None

    @staticmethod
    def _is_paddle_runtime_fatal(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(token in msg for token in [
            "could not create a primitive",
            "onednn",
            "mkldnn",
            "cuda error",
            "out of memory",
            "cudnn",
        ])

    def _perform_ocr_paddle(self, images: List[Any], batch_size: int = 1) -> Optional[str]:
        """
        Esegue OCR con PaddleOCR su una lista di immagini.
        OTTIMIZZATO: Batch processing per maggiore throughput.
        
        Args:
            images: Lista di immagini PIL o numpy arrays
            batch_size: Numero di immagini per batch (default 4)
        """
        global _PADDLE_GPU_GLOBALLY_FAILED
        ocr = self._get_paddle_ocr()
        if not ocr:
            return None

        try:
            import numpy as np
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Prepara tutte le immagini come numpy arrays
            prepared_images = []
            for image in images:
                try:
                    if hasattr(image, "mode"):
                        prepared_images.append(np.array(image))
                    else:
                        prepared_images.append(image)
                except Exception:
                    prepared_images.append(image)
            
            all_text = []
            
            # Processo in batch per efficienza
            def process_page(args):
                idx, image = args
                try:
                    result = ocr.ocr(image, cls=True)
                    lines = []
                    for block in result or []:
                        for line in block:
                            text = line[1][0] if line and len(line) > 1 else ""
                            if text:
                                lines.append(text)
                    page_text = "\n".join(lines)
                    return idx, page_text
                except Exception as e:
                    if self._is_paddle_runtime_fatal(e):
                        logger.error(f"🔴 PaddleOCR GPU crash (pagina {idx+1}): {type(e).__name__}: {e}")
                        raise RuntimeError(f"PADDLE_FATAL::{e}")
                    logger.warning(f"PaddleOCR pagina {idx+1} fallito: {e}")
                    return idx, ""
            
            # Esegui OCR in parallelo (batch_size thread)
            results = {}
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(process_page, (i, img)): i 
                    for i, img in enumerate(prepared_images)
                }
                for future in as_completed(futures):
                    try:
                        idx, page_text = future.result(timeout=30)
                        results[idx] = page_text
                    except Exception as e:
                        if self._is_paddle_runtime_fatal(e):
                            logger.error(f"🔴 PaddleOCR GPU crash (pagina): {type(e).__name__}: {e}")
                            self._paddle_ocr = None
                            # Tenta CPU-mode prima di rinunciare definitivamente
                            try:
                                from paddleocr import PaddleOCR as _PaddleOCR
                                self._paddle_ocr = _PaddleOCR(
                                    use_angle_cls=True,
                                    lang=getattr(settings, "paddleocr_lang", "it"),
                                    show_log=False,
                                    use_gpu=False,
                                    enable_mkldnn=False,
                                )
                                logger.warning("⚠️ PaddleOCR GPU instabile: riavviato in CPU mode per i prossimi documenti.")
                                _PADDLE_GPU_GLOBALLY_FAILED = True
                            except Exception:
                                self._paddle_runtime_failed = True
                                _PADDLE_GPU_GLOBALLY_FAILED = True
                                logger.warning("⚠️ PaddleOCR disabilitato per runtime instabile (fallback Tesseract).")
                            return None
                        logger.warning(f"PaddleOCR batch timeout: {e}")
            
            # Riordina risultati per indice pagina
            for i in range(len(prepared_images)):
                page_text = results.get(i, "")
                all_text.append(f"--- PAGINA {i + 1} ---\n{page_text}")

            return "\n\n".join(all_text) if all_text else None
        except Exception as e:
            if self._is_paddle_runtime_fatal(e):
                logger.error(f"🔴 PaddleOCR GPU crash (outer): {type(e).__name__}: {e}")
                self._paddle_ocr = None
                try:
                    from paddleocr import PaddleOCR as _PaddleOCR
                    self._paddle_ocr = _PaddleOCR(
                        use_angle_cls=True,
                        lang=getattr(settings, "paddleocr_lang", "it"),
                        show_log=False,
                        use_gpu=False,
                        enable_mkldnn=False,
                    )
                    logger.warning("⚠️ PaddleOCR GPU instabile: riavviato in CPU mode per i prossimi documenti.")
                    _PADDLE_GPU_GLOBALLY_FAILED = True
                except Exception:
                    self._paddle_runtime_failed = True
                    _PADDLE_GPU_GLOBALLY_FAILED = True
                    logger.warning("⚠️ PaddleOCR disabilitato per runtime instabile (fallback Tesseract).")
                return None
            logger.warning(f"PaddleOCR fallito: {e}")
            return None
    
    def _process_word(self, doc_path: Path) -> OCRResult:
        """
        Processa documenti Word (DOCX, DOC, ODT, RTF).
        Usa enhanced parser per gestione tabelle avanzata.
        """
        try:
            # 1. Prova enhanced parser (gestione tabelle migliore)
            if doc_path.suffix.lower() in ['.docx']:
                try:
                    from .enhanced_parser import get_parser
                    
                    parser = get_parser()
                    parsed = parser.parse(str(doc_path))
                    
                    if parsed.success:
                        logger.info(f"📝 Word processato (enhanced): {doc_path.name}")
                        return OCRResult(
                            text=parsed.text,
                            pages=1,
                            confidence=0.95,
                            method="enhanced_parser",
                            metadata=parsed.metadata
                        )
                except Exception as e:
                    logger.debug(f"Enhanced parser fallback per DOCX: {str(e)}")
                
                # 2. Fallback a python-docx standard
                try:
                    from docx import Document
                    doc = Document(doc_path)
                    
                    paragraphs = []
                    for para in doc.paragraphs:
                        if para.text.strip():
                            paragraphs.append(para.text)
                    
                    # Estrai anche testo dalle tabelle
                    for table in doc.tables:
                        for row in table.rows:
                            row_text = " | ".join(cell.text.strip() for cell in row.cells)
                            if row_text.strip():
                                paragraphs.append(row_text)
                    
                    text = "\n\n".join(paragraphs)
                    
                    logger.info(f"📝 Word processato (fallback): {doc_path.name}")
                    
                    return OCRResult(
                        text=text,
                        pages=1,
                        confidence=0.90,
                        method="docx_fallback",
                        metadata={"paragraphs": len(paragraphs)}
                    )
                    
                except ImportError:
                    logger.warning("⚠️ python-docx non installato. Installa: pip install python-docx")
                    return OCRResult(
                        text="",
                        pages=0,
                        confidence=0,
                        method="error",
                        errors=["python-docx non installato"]
                    )
            
            # DOC, RTF, ODT - prova conversione con altri metodi
            else:
                # Fallback: prova a leggere come testo
                try:
                    text = doc_path.read_text(encoding='utf-8', errors='ignore')
                    return OCRResult(
                        text=text,
                        pages=1,
                        confidence=0.5,
                        method="text_fallback",
                        metadata={"warning": "Formato DOC/RTF/ODT richiede conversione avanzata"}
                    )
                except:
                    return OCRResult(
                        text="",
                        pages=0,
                        confidence=0,
                        method="error",
                        errors=[f"Formato {doc_path.suffix} non leggibile direttamente. Usa DOCX."]
                    )
                    
        except Exception as e:
            logger.error(f"Errore Word: {e}")
            return OCRResult(
                text="",
                pages=0,
                confidence=0,
                method="error",
                errors=[str(e)]
            )
    
    def _process_excel(self, excel_path: Path) -> OCRResult:
        """
        Processa file Excel (XLSX, XLS).
        Usa enhanced parser per gestione multi-sheet ottimizzata.
        """
        try:
            # 1. Prova enhanced parser
            if excel_path.suffix.lower() in ['.xlsx', '.csv']:
                try:
                    from .enhanced_parser import get_parser
                    
                    parser = get_parser()
                    parsed = parser.parse(str(excel_path))
                    
                    if parsed.success:
                        logger.info(f"📊 Excel processato (enhanced): {excel_path.name}")
                        return OCRResult(
                            text=parsed.text,
                            pages=parsed.metadata.get('sheets', 1),
                            confidence=0.95,
                            method="enhanced_parser",
                            metadata=parsed.metadata
                        )
                except Exception as e:
                    logger.debug(f"Enhanced parser fallback per Excel: {str(e)}")
            
            # 2. Fallback a openpyxl standard
            from openpyxl import load_workbook
            
            wb = load_workbook(excel_path, data_only=True)
            
            all_text = []
            sheet_count = len(wb.sheetnames)
            
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                all_text.append(f"=== FOGLIO: {sheet_name} ===\n")
                
                rows_text = []
                for row in ws.iter_rows(values_only=True):
                    row_values = [str(cell) if cell is not None else "" for cell in row]
                    if any(v.strip() for v in row_values):
                        rows_text.append(" | ".join(row_values))
                
                all_text.append("\n".join(rows_text))
            
            wb.close()
            
            text = "\n\n".join(all_text)
            
            logger.info(f"📊 Excel processato (fallback): {excel_path.name} ({sheet_count} fogli)")
            
            return OCRResult(
                text=text,
                pages=sheet_count,
                confidence=0.90,
                method="xlsx_fallback",
                metadata={"sheets": wb.sheetnames if hasattr(wb, 'sheetnames') else []}
            )
            
        except ImportError:
            logger.warning("⚠️ openpyxl non installato")
            return OCRResult(
                text="",
                pages=0,
                confidence=0,
                method="error",
                errors=["openpyxl non installato"]
            )
        except Exception as e:
            logger.error(f"Errore Excel: {e}")
            return OCRResult(
                text="",
                pages=0,
                confidence=0,
                method="error",
                errors=[str(e)]
            )
    
    def _process_text(self, text_path: Path) -> OCRResult:
        """
        Processa file di testo (TXT, CSV).
        """
        try:
            # Prova diverse codifiche
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            text = None
            used_encoding = None
            
            for enc in encodings:
                try:
                    text = text_path.read_text(encoding=enc)
                    used_encoding = enc
                    break
                except UnicodeDecodeError:
                    continue
            
            if text is None:
                text = text_path.read_text(encoding='utf-8', errors='ignore')
                used_encoding = 'utf-8 (fallback)'
            
            logger.info(f"📄 Testo processato: {text_path.name} (encoding: {used_encoding})")
            
            return OCRResult(
                text=text,
                pages=1,
                confidence=1.0,
                method="text",
                metadata={"encoding": used_encoding, "size_bytes": text_path.stat().st_size}
            )
            
        except Exception as e:
            logger.error(f"Errore testo: {e}")
            return OCRResult(
                text="",
                pages=0,
                confidence=0,
                method="error",
                errors=[str(e)]
            )
    
    def _process_email(self, email_path: Path) -> OCRResult:
        """
        Processa file email (EML, MSG).
        Usa enhanced parser per MSG, gestione robusta per EML.
        """
        try:
            # 1. Prova enhanced parser (gestisce entrambi MSG ed EML)
            try:
                from .enhanced_parser import get_parser
                
                parser = get_parser()
                parsed = parser.parse(str(email_path))
                
                if parsed.success:
                    logger.info(f"📧 Email processata (enhanced): {email_path.name}")
                    return OCRResult(
                        text=parsed.text,
                        pages=1,
                        confidence=0.95,
                        method="enhanced_parser",
                        metadata=parsed.metadata
                    )
            except Exception as e:
                logger.debug(f"Enhanced parser fallback per email: {str(e)}")
            
            # 2. Fallback a gestione standard
            import email
            from email import policy
            
            if email_path.suffix.lower() == '.eml':
                with open(email_path, 'rb') as f:
                    msg = email.message_from_binary_file(f, policy=policy.default)
                
                # Estrai header
                headers = []
                headers.append(f"Da: {msg.get('From', 'N/A')}")
                headers.append(f"A: {msg.get('To', 'N/A')}")
                headers.append(f"CC: {msg.get('Cc', 'N/A')}")
                headers.append(f"Oggetto: {msg.get('Subject', 'N/A')}")
                headers.append(f"Data: {msg.get('Date', 'N/A')}")
                
                # Estrai corpo
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        if content_type == "text/plain":
                            body = part.get_content()
                            break
                        elif content_type == "text/html" and not body:
                            # Fallback a HTML se non c'è plain text
                            html_body = part.get_content()
                            # Rimuovi tag HTML basilari
                            import re
                            body = re.sub(r'<[^>]+>', ' ', html_body)
                else:
                    body = msg.get_content()
                
                # Lista allegati
                attachments = []
                if msg.is_multipart():
                    for part in msg.walk():
                        filename = part.get_filename()
                        if filename:
                            attachments.append(filename)
                
                text = "\n".join(headers)
                text += "\n\n--- CORPO ---\n\n"
                text += body
                
                if attachments:
                    text += "\n\n--- ALLEGATI ---\n"
                    for att in attachments:
                        text += f"- {att}\n"
                
                logger.info(f"📧 Email EML processata (fallback): {email_path.name}")
                
                return OCRResult(
                    text=text,
                    pages=1,
                    confidence=0.90,
                    method="eml_fallback",
                    metadata={
                        "from": msg.get('From', ''),
                        "to": msg.get('To', ''),
                        "subject": msg.get('Subject', ''),
                        "attachments": attachments
                    }
                )
            
            elif email_path.suffix.lower() == '.msg':
                # MSG richiede extract-msg o altro parser
                try:
                    import extract_msg
                    msg = extract_msg.Message(str(email_path))
                    
                    headers = [
                        f"Da: {msg.sender}",
                        f"A: {msg.to}",
                        f"Oggetto: {msg.subject}",
                        f"Data: {msg.date}",
                    ]
                    
                    text = "\n".join(headers)
                    text += "\n\n--- CORPO ---\n\n"
                    text += msg.body or ""
                    
                    attachments = [att.longFilename for att in msg.attachments]
                    if attachments:
                        text += "\n\n--- ALLEGATI ---\n"
                        for att in attachments:
                            text += f"- {att}\n"
                    
                    msg.close()
                    
                    logger.info(f"📧 Email MSG processata (fallback): {email_path.name}")
                    
                    return OCRResult(
                        text=text,
                        pages=1,
                        confidence=0.90,
                        method="msg_fallback",
                        metadata={"attachments": attachments}
                    )
                    
                except ImportError:
                    logger.warning("⚠️ extract-msg non installato. Installa: pip install extract-msg")
                    return OCRResult(
                        text="",
                        pages=0,
                        confidence=0,
                        method="error",
                        errors=["extract-msg non installato per file .msg"]
                    )
            
        except Exception as e:
            logger.error(f"Errore email: {e}")
            return OCRResult(
                text="",
                pages=0,
                confidence=0,
                method="error",
                errors=[str(e)]
            )

    def extract_pages(self, pdf_path: Path) -> List[PageContent]:
        """
        Estrae contenuto pagina per pagina con metadati.
        Utile per riferimenti precisi nel report.
        """
        pdf_path = Path(pdf_path)
        pages = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                images = page.get_images()
                
                # Semplice euristica per tabelle
                has_tables = self._detect_tables(text)
                
                pages.append(PageContent(
                    page_num=page_num + 1,
                    text=text,
                    has_images=len(images) > 0,
                    has_tables=has_tables
                ))
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Errore estrazione pagine: {e}")
        
        return pages
    
    def _detect_tables(self, text: str) -> bool:
        """Euristica semplice per rilevare tabelle"""
        lines = text.split('\n')
        # Se molte righe hanno tab o molti spazi, probabilmente c'è una tabella
        tab_lines = sum(1 for line in lines if '\t' in line or '   ' in line)
        return tab_lines > 3


# Singleton
_ocr_engine: Optional[OCREngine] = None

def get_ocr_engine() -> OCREngine:
    """Ottieni l'istanza singleton dell'OCR Engine"""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine()
    return _ocr_engine
