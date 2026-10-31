# -*- coding: utf-8 -*-
# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
Smart Document Pipeline - Flusso ottimizzato per documenti bancari

Integra:
1. Florence-2 per analisi visiva (firme, layout)
2. Routing intelligente (tabella vs testo)
3. PaddleOCR con modalità appropriata
4. Bypass OCR per PDF nativi
5. Preprocessing OpenCV condizionale
"""

# FIX: Disabilitare PIR prima di importare PaddlePaddle
# Questo deve essere PRIMA di qualsiasi import che potrebbe caricare paddle
import os
import ctypes
import sys as _sys_init
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# FIX Python 3.8+: os.add_dll_directory a livello modulo, prima di qualsiasi
# import che possa caricare paddle/CUDA. PATH non viene usato per DLL extension.
_dll_dirs_module = []
for _dp in [
    os.path.join(_sys_init.prefix, "Scripts"),
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
] + [p for p in os.environ.get("PATH", "").split(os.pathsep) if p and "cuda" in p.lower()]:
    if _dp and os.path.isdir(_dp):
        try:
            _dll_dirs_module.append(os.add_dll_directory(_dp))
        except Exception:
            pass

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import re
from threading import Lock
import contextlib
import threading
from loguru import logger
import numpy as np

from zt_legal_analysis_core.core.config import settings

_JUPYTER_KERNEL = "ipykernel" in _sys_init.modules
_jupyter_silent_io = threading.local()

if _JUPYTER_KERNEL:
    os.environ.setdefault("TQDM_DISABLE", "1")
    _builtin_print = _sys_init.modules["builtins"].print

    def _jupyter_aware_print(*args, **kwargs):
        if getattr(_jupyter_silent_io, "active", False):
            return
        _builtin_print(*args, **kwargs)

    _sys_init.modules["builtins"].print = _jupyter_aware_print


def _sp_log(level: str, message: str, *args, **kwargs) -> None:
    """Evita scritture su stderr ZMQ da worker thread Jupyter durante OCR Surya."""
    if getattr(_jupyter_silent_io, "active", False):
        return
    getattr(logger, level)(message, *args, **kwargs)


@contextlib.contextmanager
def _jupyter_worker_thread_io_guard():
    """
    In Jupyter/ipykernel, sys.stderr è un OutStream ZMQ non thread-safe.
    Surya (tqdm) e loguru che scrivono su stderr da ThreadPoolExecutor causano:
    zmq.error.ZMQError: Socket operation on non-socket

    NON sostituire sys.stderr: è process-global e, se chiuso da un worker,
    il kernel va in errore (ValueError: I/O operation on closed file).
    """
    in_worker = threading.current_thread() is not threading.main_thread()
    if not (_JUPYTER_KERNEL and in_worker):
        yield
        return

    prev_tqdm = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    _jupyter_silent_io.active = True

    _orig_tqdm = None
    _orig_auto_tqdm = None
    try:
        import tqdm as _tqdm_mod

        _orig_tqdm = _tqdm_mod.tqdm
        _orig_auto_tqdm = getattr(_tqdm_mod.auto, "tqdm", None)

        def _silent_tqdm(*args, **kwargs):
            kwargs["disable"] = True
            return _orig_tqdm(*args, **kwargs)

        _tqdm_mod.tqdm = _silent_tqdm
        if _orig_auto_tqdm is not None:
            _tqdm_mod.auto.tqdm = _silent_tqdm
    except Exception:
        pass

    try:
        yield
    finally:
        _jupyter_silent_io.active = False
        if _orig_tqdm is not None:
            try:
                import tqdm as _tqdm_mod

                _tqdm_mod.tqdm = _orig_tqdm
                if _orig_auto_tqdm is not None:
                    _tqdm_mod.auto.tqdm = _orig_auto_tqdm
            except Exception:
                pass
        if prev_tqdm is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = prev_tqdm


def _is_paddle_gpu_runtime_available() -> bool:
    """Verifica che cuDNN 8 E cublas CUDA 12 siano disponibili (richiesti da PaddleOCR 2.6.x GPU).
    Su GPU Ada Lovelace / sm_89 (L40S, RTX 4090) con paddle < 3.0 restituisce False.
    """
    if os.name != "nt":
        # Linux: controlla compute capability per Ada Lovelace (sm_89)
        try:
            import torch
            if torch.cuda.is_available():
                cc = torch.cuda.get_device_capability(0)
                if cc >= (8, 9):  # Ada Lovelace / sm_89+
                    try:
                        import paddle  # type: ignore
                        paddle_ver = tuple(int(x) for x in getattr(paddle, "__version__", "0.0.0").split(".")[:2])
                        if paddle_ver < (3, 0):
                            return False  # paddle < 3.0 non supporta sm_89
                    except Exception:
                        return False
        except Exception:
            pass
        return True
    import sys

    # Costruisce lista di cartelle dove cercare le DLL
    search_dirs = [
        os.path.join(sys.prefix, "Scripts"),   # venv/Scripts (DLL alias CUDA 12)
        r"C:\Windows\System32",
        r"C:\Windows\SysWOW64",
    ]
    # Aggiunge cartelle CUDA eventualmente presenti in PATH
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if "cuda" in p.lower() and os.path.isdir(p):
            search_dirs.append(p)

    def _dll_found(name: str) -> bool:
        # Prova ctypes (usa ordine di ricerca DLL di Windows + PATH)
        try:
            ctypes.WinDLL(name)
            return True
        except Exception:
            pass
        # Prova esplicitamente su filesystem (fallback se PATH non include la dir)
        return any(os.path.isfile(os.path.join(d, name)) for d in search_dirs)

    for dll in ("cudnn64_8.dll", "cublasLt64_12.dll"):
        if not _dll_found(dll):
            return False
    return True


class ProcessingStrategy(Enum):
    """Strategia di processing selezionata"""
    NATIVE_PDF = "native_pdf"           # PDF con testo selezionabile
    OCR_TABLE = "ocr_table"             # Scansione con tabelle (estratto conto)
    OCR_TEXT = "ocr_text"               # Scansione con testo fitto
    OCR_MIXED = "ocr_mixed"             # Documento misto
    ENHANCED_PARSER = "enhanced_parser" # Parser avanzato (PyMuPDF4LLM)


@dataclass
class SmartPipelineResult:
    """Risultato pipeline intelligente"""
    text: str
    pages: int
    confidence: float
    strategy: ProcessingStrategy
    processing_time: float = 0.0
    
    # Metadata visivo
    has_signature: bool = False
    signature_count: int = 0
    has_stamp: bool = False
    is_scanned: bool = False
    quality_score: float = 1.0
    
    # Warnings per l'utente
    warnings: List[str] = field(default_factory=list)
    
    # Metadata tecnico
    metadata: Dict[str, Any] = field(default_factory=dict)


class SmartDocumentPipeline:
    """
    Pipeline intelligente per documenti bancari.
    
    Flusso:
    1. Check PDF nativo → se sì, estrai direttamente
    2. Florence-2 analisi visiva → firme + layout
    3. Routing → PaddleOCR text/table mode
    4. Preprocessing OpenCV se qualità bassa
    5. Entity extraction con contesto arricchito
    
    Ottimizzato per RTX 4060 Ti 16GB:
    - Florence-2: ~2GB VRAM
    - PaddleOCR: ~1.5GB VRAM  
    - Qwen2.5: ~5.5GB VRAM
    - Totale picco: ~9GB (margine ampio)
    """
    
    # Soglie
    NATIVE_TEXT_MIN_LENGTH = 200       # Caratteri minimi per considerare PDF nativo
    QUALITY_THRESHOLD_PREPROCESS = 0.6  # Sotto questa qualità, applica preprocessing
    CONFIDENCE_RETRY_THRESHOLD = 0.5    # Sotto questa confidence, riprova con preprocessing
    # Soglia densità alfabetica minima per considerare il testo "reale"
    # (lettera o cifra) sul totale dei caratteri non-spazio.
    ALPHA_DENSITY_MIN = 0.35

    # Singleton Surya a livello di classe: condiviso tra TUTTE le istanze nello stesso processo.
    # Evita che worker paralleli carichino Surya N volte → race condition + OOM.
    _class_surya_models = None   # None = non ancora caricato, False = caricamento fallito
    _class_surya_lock = Lock()   # Lock di classe, condiviso tra tutte le istanze
    _class_surya_inference_lock = Lock()  # Serializza l'inference: PyTorch non è thread-safe

    def __init__(self):
        """Inizializza pipeline (lazy loading dei componenti)"""
        self._vision_analyzer = None
        self._vision_circuit_open = False
        self._vision_override_enabled: Optional[bool] = None
        self._preprocessor = None
        self._paddle_ocr = None
        self._paddle_table = None
        self._paddle_failed = False   # Evita warning ripetuti
        self._paddle_gpu_precheck_logged = False
        # Surya: usa il singleton di classe (non teniamo copia per istanza)
        
        logger.info("🚀 SmartDocumentPipeline inizializzata (lazy loading)")
    
    def _is_florence_enabled(self) -> bool:
        if self._vision_override_enabled is not None:
            return bool(self._vision_override_enabled)
        return bool(getattr(settings, "enable_florence2", False))

    def _get_vision_analyzer(self):
        """Lazy load Florence-2 analyzer (se abilitato in config)"""
        if self._vision_circuit_open:
            return None

        if not self._is_florence_enabled():
            return None
            
        if self._vision_analyzer is None:
            try:
                from .vision_analyzer import get_vision_analyzer
                self._vision_analyzer = get_vision_analyzer()
            except Exception:
                logger.warning("⚠️ Analisi visiva non disponibile")
        return self._vision_analyzer

    def _safe_analyze_page(self, image):
        """Analisi pagina protetta: apre circuito su errori runtime del backend visivo."""
        analyzer = self._get_vision_analyzer()
        if analyzer is None:
            return None
        try:
            return analyzer.analyze_page(image)
        except Exception as e:
            self._vision_circuit_open = True
            self._vision_analyzer = None
            logger.warning(f"⚠️ Analisi visiva disabilitata (circuit breaker): {str(e)[:120]}")
            return None
    
    def _get_preprocessor(self):
        """Lazy load image preprocessor"""
        if self._preprocessor is None:
            try:
                from .image_preprocessor import get_preprocessor
                self._preprocessor = get_preprocessor()
            except Exception as e:
                logger.warning(f"⚠️ Preprocessor non disponibile: {e}")
        return self._preprocessor
    
    def _is_unreadable_text(self, texts: list) -> bool:
        """Rileva se il testo estratto da PyMuPDF non è contenuto reale leggibile.

        Copre i seguenti casi:
        - Path immagine (es. "Page Data/Work/Img012.png")
        - Garbage / caratteri binari / sequenze non alfabetiche
        - Coordinate numeriche (es. layout PostScript "0 0 595 841")
        - Testo con densità alfabetica troppo bassa (< ALPHA_DENSITY_MIN)
        - Testo con righe reali insufficienti (< 3)

        Restituisce True se il testo NON è leggibile → il chiamante deve forzare OCR.
        """
        import re as _re

        # Pattern: riga che è solo un path immagine o un riferimento file
        _img_pat = _re.compile(
            r'^\s*(Page\s+)?[\w/\\ :\-\.]+\.(png|jpg|jpeg|tiff?|bmp|gif|pdf|svg|wmf|emf)\s*$',
            _re.IGNORECASE
        )
        # Pattern: riga che è solo numeri/coordinate (es. PostScript/PDF stream)
        _coord_pat = _re.compile(r'^\s*[-\d\.\s,;]+\s*$')
        # Pattern: riga con soli caratteri speciali / garbage
        _garbage_pat = _re.compile(r'^[\x00-\x1f\x7f-\xff\W]+$')

        combined = "\n".join(texts)
        lines = [ln for ln in combined.splitlines() if ln.strip()]

        if not lines:
            return True

        # Conta righe "reali" (non path immagine, non coordinate, non garbage)
        real_lines = [
            ln for ln in lines
            if not _img_pat.match(ln)
            and not _coord_pat.match(ln)
            and not _garbage_pat.match(ln)
        ]
        if len(real_lines) < 3:
            return True

        # Calcola densità alfabetica sul testo reale
        real_text = " ".join(real_lines)
        non_space = [c for c in real_text if not c.isspace()]
        if not non_space:
            return True
        alpha_ratio = sum(1 for c in non_space if c.isalpha() or c.isdigit()) / len(non_space)
        if alpha_ratio < self.ALPHA_DENSITY_MIN:
            return True

        # Calcola rapporto parole medie per riga (almeno 2 parole per riga in media)
        avg_words = sum(len(ln.split()) for ln in real_lines) / len(real_lines)
        if avg_words < 1.5:
            return True

        return False

    def _get_surya_models(self):
        """Lazy load e cache modelli Surya — singleton di classe, caricato una sola volta."""
        # Fast path — nessun lock se già caricati (o già falliti)
        if SmartDocumentPipeline._class_surya_models is not None:
            return SmartDocumentPipeline._class_surya_models if SmartDocumentPipeline._class_surya_models is not False else None
        # Double-checked locking sul lock DI CLASSE: un solo worker carica, gli altri aspettano
        with SmartDocumentPipeline._class_surya_lock:
            if SmartDocumentPipeline._class_surya_models is not None:
                return SmartDocumentPipeline._class_surya_models if SmartDocumentPipeline._class_surya_models is not False else None
            return self._load_surya_models_impl()

    def _load_surya_models_impl(self):
        """Carica Surya senza lock. Deve essere chiamato SOLO dentro self._surya_load_lock."""
        # Forza offline PRIMA di qualsiasi import Surya — evita timeout su models.datalab.to
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        rec_path = os.environ.get("RECOGNITION_MODEL_CHECKPOINT", "")
        det_path = os.environ.get("DETECTOR_MODEL_CHECKPOINT", "")
        # Fall back to well-known SageMaker model paths if env vars aren't set
        _DEFAULT_SURYA_BASE = "/home/sagemaker-user/surya_models"
        if not rec_path:
            rec_path = os.path.join(_DEFAULT_SURYA_BASE, "rec_v2")
        if not det_path:
            det_path = os.path.join(_DEFAULT_SURYA_BASE, "det_v2")
        # Verify at least one sentinel file exists — avoids cryptic errors later
        if not os.path.isdir(rec_path) or not os.path.isdir(det_path):
            logger.warning(
                f"⚠️ Surya disabilitato: path modelli non trovati "
                f"(rec={rec_path}, det={det_path}). "
                f"Imposta RECOGNITION_MODEL_CHECKPOINT / DETECTOR_MODEL_CHECKPOINT."
            )
            SmartDocumentPipeline._class_surya_models = False
            return None
        try:
            import sys as _sys
            import types as _types
            import importlib.util as _importlib_util
            import torch as _torch_pre

            # ── Patch 0a: torch uint types aggiunti in 2.3 (mancanti in 2.2.x) ──
            for _uname, _umapped in [('uint16', _torch_pre.int16), ('uint32', _torch_pre.int32), ('uint64', _torch_pre.int64)]:
                if not hasattr(_torch_pre, _uname):
                    setattr(_torch_pre, _uname, _umapped)

            # ── Patch 0b: torch.__version__ — transformers >= 4.48 richiede >= 2.4 ──
            if tuple(int(x) for x in _torch_pre.__version__.split('+')[0].split('.')[:2]) < (2, 4):
                _torch_pre.__version__ = "2.4.1+" + _torch_pre.__version__.split('+', 1)[-1]
                logger.info(f"✅ Patched torch.__version__ → {_torch_pre.__version__}")

            # ── Patch 0c: sklearn mock — compilato contro numpy 1.x, ABI mismatch con numpy 2.x ──
            # transformers 4.50+ importa sklearn.metrics.roc_curve in candidate_generator.py
            if 'sklearn' not in _sys.modules or not getattr(_sys.modules.get('sklearn'), '_zt_mock', False):
                _sklearn_mock = _types.ModuleType('sklearn')
                _sklearn_mock.__spec__ = _importlib_util.spec_from_loader('sklearn', loader=None)
                _sklearn_mock._zt_mock = True
                _metrics_mock = _types.ModuleType('sklearn.metrics')
                _metrics_mock.__spec__ = _importlib_util.spec_from_loader('sklearn.metrics', loader=None)
                _metrics_mock.roc_curve = lambda *a, **kw: ([], [], [])
                _sklearn_mock.metrics = _metrics_mock
                _sys.modules['sklearn'] = _sklearn_mock
                _sys.modules['sklearn.metrics'] = _metrics_mock
                logger.info("✅ Patched sklearn mock (numpy ABI workaround)")

            # ── Patch 0d: torch.cuda.is_bf16_supported — 'including_emulation' aggiunto in 2.4 ──
            if not getattr(_torch_pre.cuda.is_bf16_supported, '_zt_patched', False):
                _orig_bf16 = _torch_pre.cuda.is_bf16_supported
                def _bf16_compat(*_a, including_emulation=False, **_kw):
                    return _orig_bf16(*_a, **_kw)
                _bf16_compat._zt_patched = True
                _torch_pre.cuda.is_bf16_supported = _bf16_compat
                logger.info("✅ Patched torch.cuda.is_bf16_supported (including_emulation kwarg)")

            # ── Patch 0e: torch.nn.utils.rnn.pad_sequence — 'padding_side' aggiunto in 2.4 ──
            import torch.nn.utils.rnn as _rnn_mod
            if not getattr(_rnn_mod.pad_sequence, '_zt_patched', False):
                _orig_pad_seq = _rnn_mod.pad_sequence
                def _pad_seq_compat(sequences, batch_first=False, padding_value=0.0, padding_side='right'):
                    result = _orig_pad_seq(sequences, batch_first=batch_first, padding_value=padding_value)
                    if padding_side == 'left':
                        result = result.flip(dims=[1 if batch_first else 0])
                    return result
                _pad_seq_compat._zt_patched = True
                _rnn_mod.pad_sequence = _pad_seq_compat
                _torch_pre.nn.utils.rnn.pad_sequence = _pad_seq_compat
                logger.info("✅ Patched torch.nn.utils.rnn.pad_sequence (padding_side kwarg)")

            from surya.recognition import RecognitionPredictor  # type: ignore
            from surya.detection import DetectionPredictor      # type: ignore
            from surya.foundation import FoundationPredictor    # type: ignore
            # Patch SuryaDecoderConfig per Surya v0.17.1 (pad_token_id aggiunto in v0.18+)
            try:
                import json as _json
                from surya.common.surya import config as _surya_cfg
                if not hasattr(_surya_cfg.SuryaDecoderConfig, 'pad_token_id'):
                    _rec_cfg_path = os.path.join(rec_path, 'config.json')
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
                _cfg_file = os.path.join(rec_path, 'config.json')
                if os.path.exists(_cfg_file):
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
            # Device selection:
            # - TORCH_DEVICE=cuda:N → rispettato (override esplicito dal notebook)
            # - CUDA disponibile → cuda:0 (13+ GB liberi su ml.g6.12xlarge, Surya ne usa ~2)
            # - no CUDA → CPU fallback
            _env_device = os.environ.get("TORCH_DEVICE", "")
            if _env_device.startswith("cuda:"):
                device = _env_device
                logger.info(f"🎯 Surya: device da TORCH_DEVICE env: {device}")
            elif _torch_pre.cuda.is_available():
                device = "cuda:0"
                logger.info("🎯 Surya: device cuda:0")
            else:
                device = "cpu"
                logger.info("🖥️  Surya: device CPU (nessuna GPU disponibile)")
            os.environ["TORCH_DEVICE"] = device
            # Patch from_pretrained: forza low_cpu_mem_usage=False prima del load
            # Senza questo, i pesi MISSING (es. lm_head.weight) restano su device 'meta'
            try:
                import transformers as _tf
                _orig_fpt_func = _tf.PreTrainedModel.from_pretrained.__func__
                @classmethod  # type: ignore[misc]
                def _patched_fpt(cls, *a, **kw):
                    kw['low_cpu_mem_usage'] = False
                    return _orig_fpt_func(cls, *a, **kw)
                _tf.PreTrainedModel.from_pretrained = _patched_fpt
                try:
                    from surya.common.pretrained import SuryaPreTrainedModel as _SPT
                    _SPT.from_pretrained = _patched_fpt
                except Exception:
                    pass
                logger.info("✅ Patched from_pretrained: low_cpu_mem_usage=False")
            except Exception as _fpe:
                logger.warning(f"Patch from_pretrained fallita: {_fpe}")
            # ── Patch torch.empty: blocca allocazioni su device 'meta' durante il load ──
            # transformers crea i pesi MISSING (es. lm_head.weight) con
            # torch.empty(shape, device='meta'). Il peso risulta None in _parameters
            # (peso tied) → non trovato dalla BFS → crash al primo forward pass.
            # Intercettiamo torch.empty e reindirizziamo meta → cpu.
            import torch as _torch_ep
            _orig_torch_empty = _torch_ep.empty
            _orig_torch_zeros = _torch_ep.zeros
            _orig_torch_ones  = _torch_ep.ones
            _orig_torch_randn = _torch_ep.randn
            _torch_ep_patched = False
            try:
                def _no_meta_factory(orig_fn):
                    def _wrapper(*_a, **_kw):
                        if str(_kw.get('device', '')) == 'meta':
                            # Redirect meta → cpu (NOT cuda).
                            # transformers uses device='meta' legitimately inside
                            # load_state_dict() to inspect checkpoint key shapes —
                            # redirecting to cuda causes OOM on a busy GPU.
                            # The BFS fix below will move any cpu-stranded params
                            # to the target device after load.
                            _kw['device'] = 'cpu'
                        return orig_fn(*_a, **_kw)
                    return _wrapper
                _torch_ep.empty = _no_meta_factory(_orig_torch_empty)
                _torch_ep.zeros = _no_meta_factory(_orig_torch_zeros)
                _torch_ep.ones  = _no_meta_factory(_orig_torch_ones)
                _torch_ep.randn = _no_meta_factory(_orig_torch_randn)
                _torch_ep_patched = True
                logger.warning(f"✅ Patched torch.empty/zeros/ones/randn: meta → cpu")
            except Exception as _epe:
                logger.warning(f"Patch torch.empty fallita: {_epe}")
            try:
                # Release any cached-but-unused VRAM before loading multi-GB models
                try:
                    import torch as _torch_pre_load
                    if _torch_pre_load.cuda.is_available():
                        _torch_pre_load.cuda.empty_cache()
                except Exception:
                    pass
                det = DetectionPredictor(checkpoint=det_path, device=device)
                fp = FoundationPredictor(checkpoint=rec_path, device=device, attention_implementation="eager")
                rec = RecognitionPredictor(foundation_predictor=fp)
            finally:
                if _torch_ep_patched:
                    _torch_ep.empty = _orig_torch_empty
                    _torch_ep.zeros = _orig_torch_zeros
                    _torch_ep.ones  = _orig_torch_ones
                    _torch_ep.randn = _orig_torch_randn
                    logger.warning("✅ Ripristinato torch.empty/zeros/ones/randn originale")
            # lm_head.weight è MISSING dal checkpoint rec_v2 → transformers lo inizializza
            # su device 'meta' → crash a runtime "Tensor on device meta is not on the expected device".
            # BFS su TUTTI gli attributi di det/fp/rec (anche non-Module), poi
            # per ogni nn.Module trovato: manipola _parameters/_buffers interni direttamente.
            try:
                import torch as _torch
                import torch.nn as _nn
                _target_dev = _torch.device(device)
                _bfs_seen: set = set()
                _fixed_names: list = []

                def _bfs_fix(root: object) -> None:
                    queue = [root]
                    while queue:
                        cur = queue.pop(0)
                        cid = id(cur)
                        if cur is None or cid in _bfs_seen:
                            continue
                        _bfs_seen.add(cid)
                        if isinstance(cur, _nn.Module):
                            # --- fix params via internal dict ---
                            # Move any tensors on 'meta' OR on 'cpu' to target device.
                            # 'cpu' case arises because the _no_meta_factory patch
                            # redirects device='meta' → 'cpu' to avoid CUDA OOM
                            # during load_state_dict key inspection.
                            def _needs_fix(t):
                                return t is not None and t.device != _target_dev and \
                                    t.device.type in ('meta', 'cpu')
                            for pname in list(cur._parameters.keys()):
                                p = cur._parameters[pname]
                                if _needs_fix(p):
                                    cur._parameters[pname] = _nn.Parameter(
                                        _torch.zeros(p.shape, dtype=p.dtype, device=_target_dev),
                                        requires_grad=p.requires_grad,
                                    )
                                    _fixed_names.append(f"{type(cur).__name__}.{pname}")
                            for bname in list(cur._buffers.keys()):
                                b = cur._buffers[bname]
                                if _needs_fix(b):
                                    cur._buffers[bname] = _torch.zeros(
                                        b.shape, dtype=b.dtype, device=_target_dev)
                                    _fixed_names.append(f"{type(cur).__name__}.{bname}")
                            # enqueue children (PyTorch internal module tree)
                            for child in cur._modules.values():
                                if child is not None:
                                    queue.append(child)
                        # enqueue Python object attributes (covers non-Module wrappers)
                        try:
                            od = object.__getattribute__(cur, '__dict__')
                            for v in od.values():
                                if v is not None and not isinstance(
                                    v, (str, int, float, bool, bytes, type)
                                ):
                                    queue.append(v)
                        except (AttributeError, TypeError):
                            pass

                for _pred in (det, fp, rec):
                    _bfs_fix(_pred)

                print(f"[Surya meta-fix] scansionati {len(_bfs_seen)} oggetti, "
                      f"materializzati {len(_fixed_names)} tensori: {_fixed_names[:10]}")
                _sp_log("warning", f"✅ Surya meta-fix: {len(_fixed_names)} tensori → {device}: {_fixed_names[:8]}")
            except Exception as _mfe:
                import traceback as _mfetb
                _sp_log("warning", f"⚠️ Surya meta-fix fallito: {_mfe}\n{_mfetb.format_exc()}")
            SmartDocumentPipeline._class_surya_models = (det, rec)
            _sp_log("info", f"✅ Surya models caricati (device={device}, locale={det_path})")
            return SmartDocumentPipeline._class_surya_models
        except Exception as e:
            import traceback as _tb
            _sp_log("warning", f"⚠️ Surya non disponibile in SmartPipeline ({e}):\n{_tb.format_exc()}")
            SmartDocumentPipeline._class_surya_models = False
            return None

    def _ocr_surya(self, image) -> Tuple[str, float]:
        """OCR singola pagina con Surya v0.7+. Restituisce (text, confidence)."""
        text, conf, _ = self._ocr_surya_with_layout(image, page_number=1)
        return text, conf

    def _ocr_surya_with_layout(
        self, image, page_number: int = 1
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        OCR singola pagina con Surya v0.7+.
        Restituisce (text, confidence, layout_blocks).

        Sfrutta:
        - text_lines[i].bbox  → zone header/body/footer + posizione 2D per hints LLM
        - text_lines[i].confidence → filtra rumore OCR < MIN_CONF, passa confidence
          reale a _build_layout_semantic_hints
        - text_lines[i].polygon → compatibile con Surya che espone polygon invece di bbox
        """
        MIN_CONF = 0.30   # righe sotto questa soglia sono rumore → scartate
        with _jupyter_worker_thread_io_guard():
            return self._ocr_surya_with_layout_impl(image, page_number, MIN_CONF)

    def _ocr_surya_with_layout_impl(
        self, image, page_number: int, MIN_CONF: float
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # On CPU: large batches amortize fixed per-pass overhead.
        # 222 bboxes / batch_size=4  → 55 forward passes (~25s/page)
        # 222 bboxes / batch_size=32 →  7 forward passes (~3.5s/page)
        # No OOM risk on CPU (192 GB RAM). GPU path already removed.
        os.environ["RECOGNITION_BATCH_SIZE"] = "32"
        os.environ.setdefault("DETECTION_BATCH_SIZE", "1")
        # Use all available CPU cores for PyTorch ops (ml.g6.12xlarge = 48 vCPU)
        try:
            import torch as _torch_thr
            _ncpu = os.cpu_count() or 8
            _torch_thr.set_num_threads(_ncpu)
            _torch_thr.set_num_interop_threads(max(1, _ncpu // 4))
        except Exception:
            pass
        models = self._get_surya_models()
        if not models:
            return "", 0.0, []
        try:
            import numpy as np
            from PIL import Image as _PILImage
            det_predictor, rec_predictor = models
            # Normalizza: Surya si aspetta PIL Image (.mode); se arriva numpy → converti
            if hasattr(image, 'mode'):
                pil_image = image
                img_w, img_h = image.size
            else:
                pil_image = _PILImage.fromarray(np.asarray(image))
                img_h, img_w = np.asarray(image).shape[:2]
            # Ridimensiona se l'immagine è troppo grande → OOM nel vision encoder (attn O(n²))
            # A4 @ 216DPI = 2480×3508px → 17000 patch → 64GB attention. A 1200px = 3975 patch → ~200MB.
            # With busy GPU (~335 MiB free for activations) use 900px: ~2235 patches → ~110MB
            _MAX_SIDE = 900
            if max(img_w, img_h) > _MAX_SIDE:
                _scale = _MAX_SIDE / max(img_w, img_h)
                _new_w, _new_h = int(img_w * _scale), int(img_h * _scale)
                pil_image = pil_image.resize((_new_w, _new_h), _PILImage.LANCZOS)
                img_w, img_h = _new_w, _new_h
                _sp_log("debug", f"Surya: immagine ridimensionata a {img_w}x{img_h} (OOM prevention)")
            # Libera cache GPU prima dell'inference
            try:
                import torch as _torch_gc
                if _torch_gc.cuda.is_available():
                    _torch_gc.cuda.empty_cache()
            except Exception:
                pass
            # Serializza l'inference: i modelli PyTorch condivisi non sono thread-safe
            with SmartDocumentPipeline._class_surya_inference_lock:
                results = rec_predictor([pil_image], det_predictor=det_predictor)
            lines: List[str] = []
            layout_blocks: List[Dict[str, Any]] = []
            conf_sum = 0.0
            conf_count = 0
            for page_result in results:
                for ln in (getattr(page_result, "text_lines", None) or []):
                    t = getattr(ln, "text", "").strip()
                    if not t:
                        continue
                    ln_conf = float(getattr(ln, "confidence", 0.90) or 0.90)
                    if ln_conf < MIN_CONF:
                        continue   # filtra rumore
                    conf_sum += ln_conf
                    conf_count += 1
                    lines.append(t)
                    # Costruisci layout block da bbox o polygon
                    block: Optional[Dict[str, Any]] = None
                    bbox = getattr(ln, "bbox", None)
                    polygon = getattr(ln, "polygon", None)
                    if bbox is not None:
                        # bbox può essere [x0,y0,x1,y1] o BBox object
                        try:
                            if hasattr(bbox, 'x1'):
                                x0, y0, x1, y1 = bbox.x1, bbox.y1, bbox.x2, bbox.y2  # type: ignore
                            elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                                x0, y0, x1, y1 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                            else:
                                x0 = y0 = x1 = y1 = None
                            if x0 is not None:
                                bbox_norm = self._normalize_bbox(x0, y0, x1, y1, img_w, img_h)
                                block = {
                                    "page": page_number,
                                    "text": t[:240],
                                    "bbox_norm": bbox_norm,
                                    "zone": self._zone_from_bbox(bbox_norm),
                                    "confidence": round(ln_conf, 3),
                                    "source": "surya",
                                }
                        except Exception:
                            pass
                    if block is None and polygon is not None:
                        block = self._build_layout_block_from_polygon(
                            polygon=polygon,
                            text=t,
                            image_size=(img_w, img_h),
                            page_number=page_number,
                            confidence=ln_conf,
                            source="surya",
                        )
                    if block:
                        layout_blocks.append(block)
            text = "\n".join(lines)
            avg_conf = (conf_sum / conf_count) if conf_count > 0 else 0.90
            if text:
                _sp_log("debug", f"Surya OCR p{page_number}: {len(text)} chars, {len(layout_blocks)} blocks, conf={avg_conf:.0%}")
            return text, round(avg_conf, 3), layout_blocks
        except Exception as e:
            import traceback as _tb_infer
            err_s = str(e)
            _sp_log("warning", f"Surya OCR fallito in SmartPipeline: {e}\n" + _tb_infer.format_exc())
            if 'meta' in err_s.lower() and 'device' in err_s.lower():
                _sp_log("warning", "⚠️ Surya: meta-device error rilevato — Surya disabilitato per questo run.")
                SmartDocumentPipeline._class_surya_models = False
            return "", 0.0, []

    def _get_paddle_ocr(self, table_mode: bool = False):
        """
        Lazy load PaddleOCR.
        
        Args:
            table_mode: Se True, usa PP-StructureV2 per tabelle
        """
        if not getattr(settings, "enable_paddleocr", False):
            return None
        # Se già fallito, non riprovare
        if self._paddle_failed:
            return None
            
        try:
            import os
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            # Fix protobuf incompatibility con paddlepaddle 2.6.x
            os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
            # Fix per errore oneDNN + PIR su Windows
            os.environ.setdefault("FLAGS_enable_pir_api", "0")
            os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
            os.environ.setdefault("FLAGS_use_mkldnn", "0")

            # Python 3.8+: DLL search via PATH non funziona per estensioni C++.
            # Registra esplicitamente venv\Scripts (dove sono le DLL alias CUDA 12)
            # e le directory CUDA trovate nel PATH o nella posizione standard.
            import sys as _sys
            _dll_dirs = []
            _dll_search_paths = []
            _venv_scripts = os.path.join(_sys.prefix, "Scripts")
            if os.path.isdir(_venv_scripts):
                _dll_search_paths.append(_venv_scripts)
            # CUDA standard installation path (non sempre nel PATH di sistema)
            for _cuda_root in [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
            ]:
                if os.path.isdir(_cuda_root):
                    _dll_search_paths.append(_cuda_root)
            # Anche directory CUDA trovate nel PATH
            for _p in os.environ.get("PATH", "").split(os.pathsep):
                if _p and "cuda" in _p.lower() and os.path.isdir(_p):
                    _dll_search_paths.append(_p)
            for _dpath in _dll_search_paths:
                try:
                    _dll_dirs.append(os.add_dll_directory(_dpath))
                except Exception:
                    pass

            from paddleocr import PaddleOCR
            
            use_gpu = bool(getattr(settings, "paddleocr_use_gpu", True))
            if use_gpu and not _is_paddle_gpu_runtime_available():
                if not self._paddle_gpu_precheck_logged:
                    logger.warning("⚠️ CUDA 12 runtime non completo (manca cublasLt64_12.dll o cudnn64_8.dll). PaddleOCR avviato in CPU.")
                    self._paddle_gpu_precheck_logged = True
                use_gpu = False
                try:
                    setattr(settings, "paddleocr_use_gpu", False)
                except Exception:
                    pass
            
            # Funzione helper per creare PaddleOCR con fallback parametri
            def create_paddle_ocr():
                def _build(use_gpu_flag: bool):
                    try:
                        return PaddleOCR(
                            use_angle_cls=True,
                            lang=getattr(settings, "paddleocr_lang", "latin"),
                            use_gpu=use_gpu_flag,
                            show_log=False,
                            enable_mkldnn=False,   # previene crash oneDNN/MKL-DNN su Windows
                        )
                    except TypeError:
                        return PaddleOCR(
                            use_angle_cls=True,
                            lang=getattr(settings, "paddleocr_lang", "latin"),
                            show_log=False,
                        )

                if use_gpu:
                    try:
                        return _build(True)
                    except Exception as gpu_exc:
                        msg = str(gpu_exc).lower()
                        gpu_dynload_error = (
                            "cudnn64_8.dll" in msg
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
                        return _build(False)

                return _build(False)
            
            if table_mode:
                if self._paddle_table is None:
                    # Modalità tabella 
                    self._paddle_table = create_paddle_ocr()
                    logger.info("✅ PaddleOCR Table mode inizializzato")
                return self._paddle_table
            else:
                if self._paddle_ocr is None:
                    # Modalità testo standard
                    self._paddle_ocr = create_paddle_ocr()
                    logger.info("✅ PaddleOCR Text mode inizializzato")
                return self._paddle_ocr
                
        except Exception as e:
            self._paddle_failed = True
            logger.warning(f"⚠️ PaddleOCR non disponibile, uso Tesseract: {str(e)[:60]}")
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
            "could not locate",
            "cublaslt64",
            "cublas64",
            "cudart64",
        ])
    
    def process(self, file_path: Union[str, Path], use_florence: Optional[bool] = None) -> SmartPipelineResult:
        """
        Processa documento con pipeline intelligente.
        
        Args:
            file_path: Path al file (PDF, immagine, etc.)
            
        Returns:
            SmartPipelineResult con testo e metadata
        """
        start_time = time.time()
        file_path = Path(file_path)
        previous_override = self._vision_override_enabled
        self._vision_override_enabled = use_florence
        
        if not file_path.exists():
            return SmartPipelineResult(
                text="",
                pages=0,
                confidence=0,
                strategy=ProcessingStrategy.NATIVE_PDF,
                warnings=[f"File non trovato: {file_path}"]
            )
        
        logger.info(f"📄 SmartPipeline: {file_path.name}")
        
        # Determina tipo file
        ext = file_path.suffix.lower()
        
        try:
            if ext == ".pdf":
                result = self._process_pdf(file_path)
            elif ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]:
                result = self._process_image(file_path)
            else:
                result = self._fallback_standard(file_path)
        finally:
            self._vision_override_enabled = previous_override
        
        result.processing_time = time.time() - start_time
        
        logger.info(
            f"   ✅ Completato: {result.strategy.value}, "
            f"conf={result.confidence:.0%}, "
            f"time={result.processing_time:.1f}s"
        )
        
        return result
    
    def _process_pdf(self, pdf_path: Path) -> SmartPipelineResult:
        """
        Processa PDF con strategia ottimale.
        
        Flusso:
        1. Prova estrazione nativa (PDF con testo)
        2. Se fallisce → converti a immagini e analizza
        3. Florence-2 per layout e firme
        4. Route a OCR appropriato
        """
        import fitz
        
        # 1. Prova estrazione nativa
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            
            # Estrai testo nativo
            native_text = []
            has_text = False
            
            for page_num in range(min(page_count, 5)):  # Sample prime 5 pagine
                page = doc[page_num]
                text = page.get_text()
                if text and len(text.strip()) > 50:
                    native_text.append(text)
                    has_text = True
            
            # Rileva PDF con testo non leggibile (ghost text, path immagine, garbage).
            if has_text and self._is_unreadable_text(native_text):
                logger.info("   🔍 Rilevato PDF con testo non leggibile (ghost/garbage) - forzo Surya OCR")
                has_text = False  # forza path scansionato
            
            # Se PDF nativo con abbastanza testo
            if has_text and sum(len(t) for t in native_text) > self.NATIVE_TEXT_MIN_LENGTH:
                logger.info("   📄 PDF nativo rilevato - estrazione diretta")
                
                # Estrai tutto il testo
                all_text = []
                layout_blocks: List[Dict[str, Any]] = []
                for page_num in range(page_count):
                    page = doc[page_num]
                    all_text.append(f"--- PAGINA {page_num + 1} ---\n{page.get_text()}")
                    layout_blocks.extend(self._extract_native_layout_blocks(page, page_num + 1))

                layout_blocks = self._limit_layout_blocks(layout_blocks)
                layout_hints = self._build_layout_semantic_hints(layout_blocks)
                
                # Analisi visiva per firme (solo prima e ultima pagina)
                vision_result = self._analyze_pdf_pages_vision(doc, [0, page_count - 1])
                
                doc.close()
                
                return SmartPipelineResult(
                    text="\n\n".join(all_text),
                    pages=page_count,
                    confidence=0.95,
                    strategy=ProcessingStrategy.NATIVE_PDF,
                    has_signature=vision_result.get("has_signature", False),
                    signature_count=vision_result.get("signature_count", 0),
                    has_stamp=vision_result.get("has_stamp", False),
                    is_scanned=False,
                    quality_score=1.0,
                    warnings=vision_result.get("warnings", []),
                    metadata={
                        "method": "native",
                        "pages": page_count,
                        "doc_layout_blocks": layout_blocks,
                        "layout_semantic_hints": layout_hints,
                    }
                )
            
            # 2. PDF scansionato - converti a immagini
            logger.info("   🔍 PDF scansionato - avvio analisi visiva")
            
            return self._process_pdf_scanned(doc, pdf_path)
            
        except Exception as e:
            logger.error(f"Errore processing PDF: {e}")
            return SmartPipelineResult(
                text="",
                pages=0,
                confidence=0,
                strategy=ProcessingStrategy.NATIVE_PDF,
                warnings=[f"Errore: {str(e)}"]
            )
    
    def _process_pdf_scanned(self, doc, pdf_path: Path) -> SmartPipelineResult:
        """Processa PDF scansionato con pipeline completa"""
        import fitz
        from PIL import Image
        
        page_count = len(doc)
        all_text = []
        page_texts = []
        all_vision_results = []
        total_confidence = 0
        layout_blocks: List[Dict[str, Any]] = []
        strategy = ProcessingStrategy.OCR_TEXT
        warnings = []
        
        # Determina strategia dal primo gruppo di pagine
        sample_pages = min(3, page_count)
        is_table_layout = False
        
        # OTTIMIZZAZIONE: OCR Table solo per estratti conto e piani ammortamento
        # Altri documenti usano OCR Text (10x più veloce)
        filename_lower = pdf_path.name.lower()
        table_document_keywords = [
            "estratto_conto", "estratto conto", "ec_mvts", "piano_ammortamento",
            "piano ammortamento", "movimenti", "saldo", "prospetto"
        ]
        force_table_mode = any(kw in filename_lower for kw in table_document_keywords)
        force_text_mode = not force_table_mode  # Default: text mode (più veloce)

        # Ottimizzazione: se force_text_mode è già determinato dal filename (caso più comune),
        # salta il loop di campionamento — salva ~15s×3 pagine per ogni documento non-tabulare
        # Matrix(1.5,1.5) = 108 DPI: A4 → 1240×1754px → ridotto a 900px da Surya.
        # Matrix(3,3) = 216 DPI produceva 222 bbox/pagina → 7 forward pass su CPU.
        # A 108 DPI: ~55 bbox → 2 forward pass → 4× più veloce.
        _PDF_MATRIX = fitz.Matrix(1.5, 1.5)
        if not force_text_mode:
            for page_num in range(sample_pages):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=_PDF_MATRIX)
                img = Image.frombytes(
                    "RGB" if pix.alpha == 0 else "RGBA",
                    [pix.width, pix.height],
                    pix.samples
                )
                if img.mode == "RGBA":
                    img = img.convert("RGB")

                # Analisi visiva
                vision_result = self._safe_analyze_page(img)
                if vision_result:
                    all_vision_results.append(vision_result)

                    # Usa table mode solo se:
                    # 1. Il filename suggerisce documento tabulare, OPPURE
                    # 2. Vision rileva tabelle E non abbiamo forzato text mode
                    if force_table_mode or (vision_result.recommended_parser == "table" and not force_text_mode):
                        is_table_layout = True
        
        # Seleziona strategia
        if not self._get_vision_analyzer() and self._is_florence_enabled():
            warnings.append("ANALISI_VISIVA_NON_DISPONIBILE")

        if is_table_layout:
            strategy = ProcessingStrategy.OCR_TABLE
            logger.info("   📊 Layout tabulare rilevato - uso OCR Table mode")
        else:
            strategy = ProcessingStrategy.OCR_TEXT
            logger.info("   ⚡ Layout testuale - uso OCR Text mode (fast)")
        
        # ── OTTIMIZZAZIONE ESTRATTO CONTO ───────────────────────────────────────
        # Un estratto conto ha struttura ripetitiva: le pagine centrali (movimenti)
        # non aggiungono entità chiave rispetto a prima+ultima (intestatario,
        # periodo, saldo). OCR solo prima+ultima → risparmio (N-2)×~12s/pagina.
        _EC_KEYWORDS = frozenset([
            "estratto conto", "conto corrente", "movimenti", "saldo", "dare", "avere",
            "accrediti", "addebiti", "valuta competenza", "banca", "intestato a",
        ])
        _page0_precomputed: Optional[str] = None
        _is_ec = force_table_mode  # già noto se filename contiene keyword EC
        max_pages = getattr(settings, "ocr_max_pages", 50)
        pages_to_process = page_count if not max_pages or max_pages <= 0 else min(page_count, max_pages)

        if not _is_ec and pages_to_process > 4:
            # Quick OCR prima pagina per rilevare EC (costo: ~12s, risparmio: ~(N-2)×12s)
            _p0_pix = doc[0].get_pixmap(matrix=_PDF_MATRIX)
            _p0_img = Image.frombytes(
                "RGB" if _p0_pix.alpha == 0 else "RGBA",
                [_p0_pix.width, _p0_pix.height],
                _p0_pix.samples,
            )
            if _p0_img.mode == "RGBA":
                _p0_img = _p0_img.convert("RGB")
            print(f"   🔍 OCR pagina 1/{pages_to_process} [{pdf_path.name}] (rilevamento tipo)", flush=True)
            _p0_text, _, _ = self._ocr_page_with_layout(_p0_img, table_mode=False, page_number=1)
            _p0_lower = (_p0_text or "").lower()
            if sum(1 for kw in _EC_KEYWORDS if kw in _p0_lower) >= 2:
                _is_ec = True
                _page0_precomputed = _p0_text
                logger.info(
                    f"   ⚡ Estratto conto rilevato da pg.1 — OCR solo prima+ultima "
                    f"({pages_to_process}p → 2p, risparmio ≈{(pages_to_process - 2) * 12}s)"
                )

        # Lista pagine: solo prima+ultima per EC (>4 pg), tutte le altre altrimenti
        if _is_ec and pages_to_process > 4:
            _last_idx = pages_to_process - 1
            _page_indices = [0] if _last_idx == 0 else [0, _last_idx]
        else:
            _page_indices = list(range(pages_to_process))

        for page_num in _page_indices:
            page = doc[page_num]
            pix = page.get_pixmap(matrix=_PDF_MATRIX)
            img = Image.frombytes(
                "RGB" if pix.alpha == 0 else "RGBA",
                [pix.width, pix.height],
                pix.samples
            )
            if img.mode == "RGBA":
                img = img.convert("RGB")
            
            # Analisi visiva se non già fatta
            if page_num >= sample_pages:
                vision_result = self._safe_analyze_page(img)
                if vision_result:
                    all_vision_results.append(vision_result)
            else:
                vision_result = all_vision_results[page_num] if page_num < len(all_vision_results) else None
            
            # Preprocessing per documenti scansionati
            # Forzato se config abilita preprocessing contrasto/grayscale 
            quality_score = vision_result.quality_score if vision_result else 0.7
            force_preprocess = getattr(settings, 'ocr_preprocess_contrast', True) or \
                               getattr(settings, 'ocr_preprocess_grayscale', True)
            
            if force_preprocess or quality_score < self.QUALITY_THRESHOLD_PREPROCESS:
                preprocessor = self._get_preprocessor()
                if preprocessor:
                    if page_num == 0:
                        logger.info("   🔧 Preprocessing attivo (contrasto + scala grigi)")
                    img = Image.fromarray(
                        preprocessor.preprocess_for_ocr(img, is_table=is_table_layout)
                    )
            
            # Fast path per OCR_TEXT: tenta estrazione nativa prima di Surya
            # (più veloce, uguale o migliore qualità per PDF con layer testuale parziale)
            page_text = ""
            page_confidence = 0.0
            page_layout_blocks: List[Dict[str, Any]] = []
            # Usa testo pre-calcolato della pg.0 (rilevamento EC anticipato — evita doppio OCR)
            if page_num == 0 and _page0_precomputed is not None:
                page_text = _page0_precomputed
                page_confidence = 0.90
            elif strategy == ProcessingStrategy.OCR_TEXT:
                try:
                    _native_blocks = doc[page_num].get_text("blocks")
                    _native_text = "\n".join(
                        b[4].strip() for b in _native_blocks
                        if isinstance(b, (list, tuple)) and len(b) > 4 and isinstance(b[4], str) and b[4].strip()
                    )
                    if len(_native_text.strip()) > 50 and not self._is_unreadable_text([_native_text]):
                        page_text = _native_text
                        page_confidence = 0.95
                except Exception:
                    pass

            _used_ocr = False
            if not page_text:
                # Native text insufficiente: usa OCR (Surya → PaddleOCR → Tesseract)
                print(f"   🔍 OCR pagina {page_num + 1}/{pages_to_process} [{pdf_path.name}]", flush=True)
                page_text, page_confidence, page_layout_blocks = self._ocr_page_with_layout(
                    img,
                    table_mode=is_table_layout,
                    page_number=page_num + 1,
                )
                _used_ocr = True
                _n_chars = len(page_text or "")
                print(
                    f"      ↳ Surya: {_n_chars} char, conf={page_confidence:.2f}"
                    + (" ⚠️ VUOTO" if _n_chars == 0 else ""),
                    flush=True
                )
            if page_layout_blocks:
                layout_blocks.extend(page_layout_blocks)
            
            all_text.append(f"--- PAGINA {page_num + 1} ---\n{page_text or ''}")
            page_texts.append({
                "page": page_num + 1,
                "text": page_text or "",
                "confidence": page_confidence
            })
            if page_text:
                total_confidence += page_confidence
        
        doc.close()

        # --- Riepilogo diagnostico per documento ---
        _n_processed = len(_page_indices)
        _ocr_pages = [p for p in page_texts if p["confidence"] < 0.99]  # heuristic: native = 0.95 fixed
        _total_chars = sum(len(p["text"]) for p in page_texts)
        _empty_pages = sum(1 for p in page_texts if len(p["text"].strip()) == 0)
        print(
            f"   📊 [{pdf_path.name}] {_n_processed}/{pages_to_process}p | "
            f"chars totali={_total_chars} | "
            f"pagine vuote={_empty_pages}/{_n_processed} | "
            f"conf_media={total_confidence/max(_n_processed,1):.2f}",
            flush=True
        )
        if _empty_pages > _n_processed * 0.5:
            print(f"   ⚠️  [{pdf_path.name}] ATTENZIONE: più del 50% delle pagine è vuota — verificare Surya", flush=True)

        # Aggrega risultati visivi
        vision_summary = self._aggregate_vision_results(all_vision_results)

        avg_confidence = total_confidence / _n_processed if _n_processed > 0 else 0
        
        # Unisci testo
        final_text = "\n\n".join(all_text)
        layout_blocks = self._limit_layout_blocks(layout_blocks)
        layout_hints = self._build_layout_semantic_hints(layout_blocks)
        
        # Pulizia OCR con LLM se abilitata (con fast-path per alta confidence)
        if getattr(settings, 'ocr_cleanup_enabled', True) and final_text:
            try:
                from zt_legal_analysis_core.agents.document_processor.ocr_cleaner import get_ocr_cleaner
                cleaner = get_ocr_cleaner()
                # Passa avg_confidence per fast-path: se alta, salta pulizia LLM
                if avg_confidence >= 0.85:
                    logger.info(f"   ⚡ OCR confidence {avg_confidence:.1%} - fast-path (solo regex)")
                else:
                    logger.info("   🧹 Pulizia testo OCR con LLM...")
                final_text = cleaner.clean(final_text, use_llm=True, ocr_confidence=avg_confidence)
            except Exception as e:
                logger.warning(f"   ⚠️ Pulizia OCR fallita: {e}")
        
        return SmartPipelineResult(
            text=final_text,
            pages=page_count,
            confidence=avg_confidence,
            strategy=strategy,
            has_signature=vision_summary.get("has_signature", False),
            signature_count=vision_summary.get("signature_count", 0),
            has_stamp=vision_summary.get("has_stamp", False),
            is_scanned=True,
            quality_score=vision_summary.get("avg_quality", 0.7),
            warnings=vision_summary.get("warnings", []) + warnings,
            metadata={
                "method": "smart_pipeline",
                "ocr_mode": "table" if is_table_layout else "text",
                "pages_processed": pages_to_process,
                "ocr_cleaned": True,
                "page_texts": page_texts,
                "doc_layout_blocks": layout_blocks,
                "layout_semantic_hints": layout_hints,
            }
        )
    
    def _process_image(self, image_path: Path) -> SmartPipelineResult:
        """Processa singola immagine"""
        from PIL import Image
        
        img = Image.open(image_path)
        if img.mode == "RGBA":
            img = img.convert("RGB")
        
        # Analisi visiva
        vision_result = None
        is_table = False
        
        vision_result = self._safe_analyze_page(img)
        if vision_result:
            is_table = vision_result.recommended_parser == "table"
        elif self._is_florence_enabled():
            warnings = ["ANALISI_VISIVA_NON_DISPONIBILE"]
        
        # Preprocessing se necessario
        quality = vision_result.quality_score if vision_result else 0.7
        if quality < self.QUALITY_THRESHOLD_PREPROCESS:
            preprocessor = self._get_preprocessor()
            if preprocessor:
                img = Image.fromarray(preprocessor.preprocess_for_ocr(img, is_table=is_table))
        
        # OCR
        text, confidence, layout_blocks = self._ocr_page_with_layout(
            img,
            table_mode=is_table,
            page_number=1,
        )
        
        # Retry con preprocessing se confidence bassa
        if confidence < self.CONFIDENCE_RETRY_THRESHOLD:
            preprocessor = self._get_preprocessor()
            if preprocessor:
                logger.info("   🔄 Confidence bassa, retry con preprocessing")
                original = Image.open(image_path).convert("RGB")
                preprocessed = preprocessor.preprocess_for_ocr(original, is_table=is_table)
                text_retry, conf_retry, retry_layout_blocks = self._ocr_page_with_layout(
                    Image.fromarray(preprocessed), 
                    table_mode=is_table,
                    page_number=1,
                )
                if conf_retry > confidence:
                    text, confidence = text_retry, conf_retry
                    layout_blocks = retry_layout_blocks
        
        strategy = ProcessingStrategy.OCR_TABLE if is_table else ProcessingStrategy.OCR_TEXT
        layout_blocks = self._limit_layout_blocks(layout_blocks)
        layout_hints = self._build_layout_semantic_hints(layout_blocks)
        
        return SmartPipelineResult(
            text=text,
            pages=1,
            confidence=confidence,
            strategy=strategy,
            has_signature=vision_result.signature_status.value in ["present", "multiple"] if vision_result else False,
            signature_count=vision_result.signature_count if vision_result else 0,
            has_stamp=vision_result.has_stamp if vision_result else False,
            is_scanned=True,
            quality_score=quality,
            warnings=warnings if 'warnings' in locals() else [],
            metadata={
                "method": "image_ocr",
                "doc_layout_blocks": layout_blocks,
                "layout_semantic_hints": layout_hints,
            }
        )
    
    def _ocr_page(
        self, 
        image, 
        table_mode: bool = False
    ) -> Tuple[str, float]:
        text, confidence, _ = self._ocr_page_with_layout(image=image, table_mode=table_mode, page_number=1)
        return text, confidence

    def _ocr_page_with_layout(
        self,
        image,
        table_mode: bool = False,
        page_number: int = 1,
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        """
        Esegue OCR su una pagina.
        
        Args:
            image: PIL Image
            table_mode: Se True, usa modalità tabella
            
        Returns:
            (text, confidence, layout_blocks)
        """
        # 1. Prova Surya OCR (transformer-based, compatibile sm_89, nessun crash cuDNN)
        # _ocr_surya_with_layout restituisce anche i layout_blocks dai bbox Surya
        # che alimentano _build_layout_semantic_hints → hints strutturali per Qwen
        surya_text, surya_conf, surya_blocks = self._ocr_surya_with_layout(image, page_number=page_number)
        # Rifiuta risultati Surya con < 50 chars: probabile rumore/fallimento riconoscimento
        # In quel caso Tesseract darà risultati migliori
        if surya_text and surya_conf >= 0.50 and len(surya_text.strip()) >= 50:
            return surya_text, surya_conf, surya_blocks

        # 2. Fallback a PaddleOCR (solo se enable_paddleocr=True)
        ocr = self._get_paddle_ocr(table_mode=table_mode)
        if ocr is not None:
            try:
                import numpy as np
                img_array = np.array(image)
                layout_blocks: List[Dict[str, Any]] = []
                
                # PaddleOCR 2.7.x usa ocr(), 3.4.0 usa predict()
                if hasattr(ocr, 'ocr'):
                    result = ocr.ocr(img_array, cls=True)
                elif hasattr(ocr, 'predict'):
                    result = ocr.predict(img_array)
                else:
                    raise AttributeError("PaddleOCR: né ocr() né predict() disponibili")
                
                lines = []
                
                # Parse risultato PaddleOCR 2.7.x: [[box, (text, conf)], ...]
                if result and isinstance(result, list):
                    for page_result in result:
                        if page_result is None:
                            continue
                        for item in page_result:
                            if isinstance(item, (list, tuple)) and len(item) >= 2:
                                box = item[0]
                                text_info = item[1]
                                extracted_text = ""
                                extracted_conf = 0.0
                                if isinstance(text_info, tuple) and len(text_info) >= 1:
                                    extracted_text = str(text_info[0])
                                    if len(text_info) >= 2:
                                        try:
                                            extracted_conf = float(text_info[1])
                                        except Exception:
                                            extracted_conf = 0.0
                                elif isinstance(text_info, str):
                                    extracted_text = text_info

                                if extracted_text:
                                    lines.append(extracted_text)
                                    block = self._build_layout_block_from_polygon(
                                        polygon=box,
                                        text=extracted_text,
                                        image_size=image.size,
                                        page_number=page_number,
                                        confidence=extracted_conf,
                                        source="paddleocr",
                                    )
                                    if block:
                                        layout_blocks.append(block)
                
                if lines:
                    return "\n".join(lines), 0.85, layout_blocks
                    
            except Exception as e:
                if self._is_paddle_runtime_fatal(e):
                    self._paddle_ocr = None
                    self._paddle_table = None
                    # Tenta CPU-mode prima di rinunciare definitivamente
                    try:
                        from paddleocr import PaddleOCR as _PaddleOCR
                        _cpu = _PaddleOCR(
                            use_angle_cls=True,
                            lang=getattr(settings, "paddleocr_lang", "it"),
                            show_log=False,
                            use_gpu=False,
                            enable_mkldnn=False,
                        )
                        self._paddle_ocr = _cpu
                        logger.warning("⚠️ PaddleOCR GPU instabile: riavviato in CPU mode per i prossimi documenti.")
                    except Exception:
                        self._paddle_failed = True
                        logger.warning("⚠️ PaddleOCR disabilitato per runtime instabile (fallback Tesseract).")
                else:
                    logger.warning(f"PaddleOCR fallito: {e}")
        
        # Fallback a Tesseract
        text, confidence = self._ocr_tesseract(image)
        return text, confidence, []
    
    def _ocr_tesseract(self, image) -> Tuple[str, float]:
        """Fallback OCR con Tesseract"""
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
            
            text = pytesseract.image_to_string(
                image,
                lang=settings.ocr_language,
                config='--oem 3 --psm 6'
            )
            
            return text, 0.70
            
        except Exception as e:
            logger.warning(f"Tesseract fallito: {e}")
            return "", 0.0

    def _extract_native_layout_blocks(self, page, page_number: int) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        try:
            page_w = float(page.rect.width) or 1.0
            page_h = float(page.rect.height) or 1.0
            for item in page.get_text("blocks") or []:
                if not isinstance(item, (list, tuple)) or len(item) < 5:
                    continue
                x0, y0, x1, y1, text = item[0], item[1], item[2], item[3], item[4]
                text = str(text or "").strip()
                if not text:
                    continue
                bbox_norm = self._normalize_bbox(x0, y0, x1, y1, page_w, page_h)
                block = {
                    "page": int(page_number),
                    "text": text[:240],
                    "bbox_norm": bbox_norm,
                    "zone": self._zone_from_bbox(bbox_norm),
                    "confidence": 1.0,
                    "source": "pymupdf",
                }
                blocks.append(block)
        except Exception as e:
            logger.debug(f"Layout nativo non disponibile pagina {page_number}: {e}")
        return blocks

    def _build_layout_block_from_polygon(
        self,
        polygon,
        text: str,
        image_size: Tuple[int, int],
        page_number: int,
        confidence: float,
        source: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            if not polygon:
                return None
            points = []
            for point in polygon:
                if not isinstance(point, (list, tuple)) or len(point) < 2:
                    continue
                points.append((float(point[0]), float(point[1])))
            if not points:
                return None

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            w = float((image_size or (1, 1))[0]) or 1.0
            h = float((image_size or (1, 1))[1]) or 1.0
            bbox_norm = self._normalize_bbox(min(xs), min(ys), max(xs), max(ys), w, h)

            clean_text = str(text or "").strip()
            if not clean_text:
                return None

            return {
                "page": int(page_number),
                "text": clean_text[:240],
                "bbox_norm": bbox_norm,
                "zone": self._zone_from_bbox(bbox_norm),
                "confidence": float(confidence or 0.0),
                "source": source,
            }
        except Exception:
            return None

    def _normalize_bbox(self, x0: float, y0: float, x1: float, y1: float, width: float, height: float) -> List[float]:
        w = max(float(width or 1.0), 1.0)
        h = max(float(height or 1.0), 1.0)
        nx0 = max(0.0, min(1.0, float(x0) / w))
        ny0 = max(0.0, min(1.0, float(y0) / h))
        nx1 = max(0.0, min(1.0, float(x1) / w))
        ny1 = max(0.0, min(1.0, float(y1) / h))
        return [round(nx0, 4), round(ny0, 4), round(nx1, 4), round(ny1, 4)]

    def _zone_from_bbox(self, bbox_norm: List[float]) -> str:
        if not isinstance(bbox_norm, list) or len(bbox_norm) < 4:
            return "body"
        y0 = float(bbox_norm[1])
        y1 = float(bbox_norm[3])
        if y0 <= 0.18:
            return "header"
        if y1 >= 0.86:
            return "footer"
        return "body"

    def _limit_layout_blocks(self, blocks: List[Dict[str, Any]], limit: int = 240) -> List[Dict[str, Any]]:
        if not isinstance(blocks, list):
            return []
        cleaned = [b for b in blocks if isinstance(b, dict) and str(b.get("text", "")).strip()]
        return cleaned[:limit]

    def _build_layout_semantic_hints(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        hints: Dict[str, Any] = {
            "heuristics_version": "position-aware-v1",
            "title_candidates": [],
            "subject_candidates": [],
            "detected_title": "",
            "detected_subject": "",
            "layout_profile": "generico",
            "profile_scores": {},
            "npl_signal_counts": {},
            "detected_legal_refs": [],
            "legal_sections": {
                "parti": [],
                "petitum": [],
                "dispositivo": [],
                "allegati": [],
            },
            "legal_section_presence": {
                "parti": False,
                "petitum": False,
                "dispositivo": False,
                "allegati": False,
            },
        }
        if not isinstance(blocks, list) or not blocks:
            return hints

        title_candidates: List[Tuple[float, str]] = []
        subject_candidates: List[Tuple[float, str]] = []
        legal_refs: List[str] = []

        profile_scores: Dict[str, int] = {
            "procedurale": 0,
            "contrattuale": 0,
            "contabile": 0,
            "immobiliare": 0,
            "societario": 0,
            "comunicazione": 0,
        }

        signal_patterns: Dict[str, str] = {
            "decreto_ingiuntivo": r"\bdecreto\s+ingiuntivo\b",
            "precetto": r"\bprecetto\b",
            "pignoramento": r"\bpignorament",
            "fallimento": r"\bfalliment",
            "concordato": r"\bconcordat",
            "sovraindebitamento": r"\bsovraindebit",
            "tribunale": r"\btribunal",
            "fideiussione": r"\bfideiuss",
            "mutuo": r"\bmutuo\b",
            "contratto": r"\bcontratt",
            "cessione_credito": r"\bcession[ea].*credit",
            "estratto_conto": r"\bestratto\s+conto\b",
            "centrale_rischi": r"\bcentrale\s+risch",
            "art_50_tub": r"art\.?\s*50\s*tub",
            "visura_catastale": r"\bvisura\s+catast",
            "visura_ipotecaria": r"\bvisura\s+ipotec",
            "visura_camerale": r"\bvisura\s+camer",
            "perizia": r"\bperizi",
            "saldo_stralcio": r"\bsaldo\s+e\s+stralcio\b",
            "piano_rientro": r"\bpiano\s+di\s+rientro\b",
            "notifica": r"\bnotific",
            "pec": r"\bpec\b",
        }
        signal_counts: Dict[str, int] = {k: 0 for k in signal_patterns}

        for block in blocks:
            text = str(block.get("text", "")).strip()
            if not text:
                continue

            page = int(block.get("page", 1) or 1)
            zone = str(block.get("zone", "body"))
            bbox = block.get("bbox_norm", [0.0, 0.0, 1.0, 1.0])
            lower = text.lower()

            is_first_page = page == 1
            is_header = zone == "header"

            if re.match(r"^\s*(oggetto|subject|rif\.|riferimento)\s*[:\-]", lower):
                subj = re.sub(r"^\s*(oggetto|subject|rif\.|riferimento)\s*[:\-]\s*", "", text, flags=re.IGNORECASE).strip()
                score = 3.0
                if is_first_page:
                    score += 0.5
                if is_header:
                    score += 0.5
                subject_candidates.append((score, subj or text))

            if is_first_page and is_header and 6 <= len(text) <= 160:
                upper_chars = [c for c in text if c.isalpha()]
                upper_ratio = (sum(1 for c in upper_chars if c.isupper()) / len(upper_chars)) if upper_chars else 0.0
                keyword_hit = any(k in lower for k in ["atto", "decreto", "contratto", "precetto", "ricorso", "cessione"])
                line_height = (float(bbox[3]) - float(bbox[1])) if isinstance(bbox, list) and len(bbox) >= 4 else 0.0
                if upper_ratio >= 0.45 or keyword_hit or line_height >= 0.035:
                    score = 1.0
                    if upper_ratio >= 0.65:
                        score += 0.8
                    if keyword_hit:
                        score += 0.6
                    if line_height >= 0.04:
                        score += 0.4
                    title_candidates.append((score, text))

            if any(tok in lower for tok in ["tribunale", "r.g.", "rg", "decreto", "precetto", "pignor"]):
                profile_scores["procedurale"] += 1
            if any(tok in lower for tok in ["fideiuss", "contratt", "mutuo", "garante", "cessione"]):
                profile_scores["contrattuale"] += 1
            if any(tok in lower for tok in ["estratto conto", "centrale rischi", "art. 50", "art 50", "saldo", "interessi", "tasso"]):
                profile_scores["contabile"] += 1
            if any(tok in lower for tok in ["visura catast", "visura ipotec", "perizia", "planimetr", "ipoteca"]):
                profile_scores["immobiliare"] += 1
            if any(tok in lower for tok in ["visura camer", "bilancio", "c.c.i.a.a", "registro imprese"]):
                profile_scores["societario"] += 1
            if any(tok in lower for tok in ["oggetto", "spett.le", "gentile", "pec", "raccomandata"]):
                profile_scores["comunicazione"] += 1

            for signal_name, pattern in signal_patterns.items():
                if re.search(pattern, lower):
                    signal_counts[signal_name] += 1
                    if len(legal_refs) < 8 and is_first_page:
                        legal_refs.append(text[:180])

        title_candidates.sort(key=lambda x: x[0], reverse=True)
        subject_candidates.sort(key=lambda x: x[0], reverse=True)

        hints["title_candidates"] = [t for _, t in title_candidates[:3]]
        hints["subject_candidates"] = [s for _, s in subject_candidates[:3]]
        hints["detected_title"] = hints["title_candidates"][0] if hints["title_candidates"] else ""
        hints["detected_subject"] = hints["subject_candidates"][0] if hints["subject_candidates"] else ""

        best_profile = max(profile_scores.items(), key=lambda kv: kv[1]) if profile_scores else ("generico", 0)
        hints["layout_profile"] = best_profile[0] if best_profile[1] >= 2 else "generico"
        hints["profile_scores"] = profile_scores
        hints["npl_signal_counts"] = {k: v for k, v in signal_counts.items() if v > 0}
        hints["detected_legal_refs"] = legal_refs

        legal_sections = self._extract_legal_sections_from_layout(blocks)
        hints["legal_sections"] = legal_sections
        hints["legal_section_presence"] = {
            "parti": len(legal_sections.get("parti", [])) > 0,
            "petitum": len(legal_sections.get("petitum", [])) > 0,
            "dispositivo": len(legal_sections.get("dispositivo", [])) > 0,
            "allegati": len(legal_sections.get("allegati", [])) > 0,
        }
        return hints

    def _extract_legal_sections_from_layout(self, blocks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        sections: Dict[str, List[Dict[str, Any]]] = {
            "parti": [],
            "petitum": [],
            "dispositivo": [],
            "allegati": [],
        }
        if not isinstance(blocks, list) or not blocks:
            return sections

        pattern_map: Dict[str, List[str]] = {
            "parti": [
                r"\bcreditor[ei]\b", r"\bdebitor[ei]\b", r"\bopponent[ei]\b", r"\bconvenut[oa]\b",
                r"\bricorrent[ei]\b", r"\bintimat[oa]\b", r"\bpart[ie]\b", r"\bcodice\s+fiscale\b",
                r"\bpartita\s+iva\b", r"\bresident[ea]\b",
            ],
            "petitum": [
                r"\bsi\s+chiede\b", r"\bchiede\s+che\b", r"\bdomanda\b", r"\brichiede\b",
                r"\bingiung[ea]\b", r"\bcondannar[ea]\b", r"\bpagamento\s+di\b", r"\bsomma\s+di\b",
                r"\binteressi\b", r"\bspese\b",
            ],
            "dispositivo": [
                r"\bp\.?q\.?m\.?\b", r"\bper\s+questi\s+motivi\b", r"\bil\s+tribunale\b",
                r"\baccoglie\b", r"\brigetta\b", r"\bordina\b", r"\bdispone\b", r"\bdichiara\b",
                r"\bdecreto\s+ingiuntivo\b", r"\bformula\s+esecutiva\b",
            ],
            "allegati": [
                r"\ballegat[io]\b", r"\belenco\s+allegat[io]\b", r"\bdocument[io]\s+allegat[io]\b",
                r"\bsi\s+allega\b", r"\ball\.?\s*\d+\b", r"\bdoc\.?\s*\d+\b",
            ],
        }

        scored: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {
            "parti": [],
            "petitum": [],
            "dispositivo": [],
            "allegati": [],
        }

        for block in blocks:
            if not isinstance(block, dict):
                continue
            text = str(block.get("text", "") or "").strip()
            if not text:
                continue

            lower = text.lower()
            page = int(block.get("page", 1) or 1)
            zone = str(block.get("zone", "body") or "body")

            for section, patterns in pattern_map.items():
                hits = sum(1 for pattern in patterns if re.search(pattern, lower))
                if hits <= 0:
                    continue

                score = float(hits)
                if section in {"parti", "petitum"} and page <= 2:
                    score += 0.8
                if section == "dispositivo" and page >= 2:
                    score += 0.6
                if section == "allegati" and zone in {"footer", "body"}:
                    score += 0.4
                if zone == "header" and section in {"parti", "petitum"}:
                    score += 0.3

                scored[section].append((
                    score,
                    {
                        "page": page,
                        "zone": zone,
                        "text": text[:220],
                        "match_count": hits,
                    },
                ))

        for section in sections:
            ranked = sorted(scored[section], key=lambda x: x[0], reverse=True)
            unique: List[Dict[str, Any]] = []
            seen = set()
            for _, item in ranked:
                key = (item.get("page"), item.get("text"))
                if key in seen:
                    continue
                seen.add(key)
                unique.append(item)
                if len(unique) >= 6:
                    break
            sections[section] = unique

        return sections
    
    def _html_table_to_text(self, html: str) -> str:
        """Converte HTML tabella semplice a testo"""
        import re
        
        # Rimuovi tag HTML mantenendo struttura
        text = re.sub(r'<tr[^>]*>', '\n', html)
        text = re.sub(r'<td[^>]*>', ' | ', text)
        text = re.sub(r'<th[^>]*>', ' | ', text)
        text = re.sub(r'</t[dhr]>', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _analyze_pdf_pages_vision(
        self, 
        doc, 
        page_indices: List[int]
    ) -> Dict[str, Any]:
        """Analizza pagine specifiche per firme e timbri"""
        import fitz
        from PIL import Image
        
        if self._get_vision_analyzer() is None:
            return {"has_signature": False, "signature_count": 0, "has_stamp": False, "warnings": []}
        
        results = []
        for page_num in page_indices:
            if page_num >= len(doc):
                continue
            
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img = Image.frombytes(
                "RGB" if pix.alpha == 0 else "RGBA",
                [pix.width, pix.height],
                pix.samples
            )
            if img.mode == "RGBA":
                img = img.convert("RGB")
            
            vision_result = self._safe_analyze_page(img)
            if vision_result:
                results.append(vision_result)
        
        return self._aggregate_vision_results(results)
    
    def _aggregate_vision_results(
        self, 
        results: List
    ) -> Dict[str, Any]:
        """Aggrega risultati analisi visiva da più pagine"""
        if not results:
            return {
                "has_signature": False,
                "signature_count": 0,
                "has_stamp": False,
                "avg_quality": 0.7,
                "warnings": []
            }
        
        total_signatures = sum(r.signature_count for r in results if r)
        has_stamp = any(r.has_stamp for r in results if r)
        avg_quality = sum(r.quality_score for r in results if r) / len(results)
        
        warnings = []
        if total_signatures == 0:
            warnings.append("⚠️ FIRMA ASSENTE - Documento potenzialmente non valido")
        elif total_signatures == 1 and len(results) > 1:
            warnings.append("⚠️ Solo 1 firma rilevata - Verificare clausole vessatorie")
        
        low_quality = [i+1 for i, r in enumerate(results) if r and r.quality_score < 0.6]
        if low_quality:
            warnings.append(f"⚠️ Qualità bassa su pagine: {low_quality}")
        
        return {
            "has_signature": total_signatures > 0,
            "signature_count": total_signatures,
            "has_stamp": has_stamp,
            "avg_quality": avg_quality,
            "warnings": warnings
        }
    
    def _fallback_standard(self, file_path: Path) -> SmartPipelineResult:
        """Fallback all'OCR Engine standard per formati non gestiti"""
        try:
            from .ocr_engine import OCREngine

            # Riusa la stessa istanza per non perdere il flag _paddle_runtime_failed
            if not hasattr(self, "_fallback_ocr_engine") or self._fallback_ocr_engine is None:
                self._fallback_ocr_engine = OCREngine()
            engine = self._fallback_ocr_engine
            result = engine.extract_text(str(file_path))
            
            return SmartPipelineResult(
                text=result.get("text", ""),
                pages=result.get("pages", 1),
                confidence=result.get("confidence", 0.5),
                strategy=ProcessingStrategy.ENHANCED_PARSER,
                metadata={"method": "fallback_standard"}
            )
            
        except Exception as e:
            logger.error(f"Fallback standard fallito: {e}")
            return SmartPipelineResult(
                text="",
                pages=0,
                confidence=0,
                strategy=ProcessingStrategy.NATIVE_PDF,
                warnings=[f"Errore: {str(e)}"]
            )
    
    def unload_models(self):
        """Scarica modelli dalla memoria per liberare VRAM"""
        if self._vision_analyzer:
            self._vision_analyzer.unload()
            self._vision_analyzer = None
        
        # PaddleOCR non ha unload esplicito, ma possiamo eliminare riferimenti
        self._paddle_ocr = None
        self._paddle_table = None
        
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        
        logger.info("🗑️ Modelli OCR (Florence-2, EasyOCR, PaddleOCR) scaricati dalla VRAM")


# Singleton
_pipeline: Optional[SmartDocumentPipeline] = None


def get_smart_pipeline() -> SmartDocumentPipeline:
    """Ottiene istanza singleton della pipeline"""
    global _pipeline
    if _pipeline is None:
        _pipeline = SmartDocumentPipeline()
    return _pipeline
