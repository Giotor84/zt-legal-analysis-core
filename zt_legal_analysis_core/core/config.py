# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
Configurazione centralizzata per l'Agente Legale Intelligente
"""

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """
    Configurazione dell'applicazione.
    I valori vengono letti da variabili d'ambiente o dal file .env
    """
    
    # === LLM Configuration - RTX 16.9GB (upgrade 2026-03-08)
    # Sintesi: mistral-nemo:12b (primario) | qwen72b:latest (fallback)
    # Estrazione: deepseek-coder-v2:16b-lite-instruct (structured JSON output)
    # Modelli sequenziali: uno solo in VRAM per volta, nessuno swap contemporaneo
    ollama_model: str = Field(default="qwen72b:latest", description="Modello LLM principale")
    
    # Modelli specializzati per agent
    entity_extractor_model: str = Field(default="qwen72b:latest", description="Entity extraction JSON — Qwen2.5-14B-Instruct")
    allow_llm_entity_extraction: bool = Field(default=True, description="Usa LLM per estrazione entità (importi, date, soggetti)")  # CRITICAL FIX: Must be True for importi extraction!
    allow_llm_doc_curation: bool = Field(default=True, description="Usa LLM per classificazione/cernita documenti rilevanti")
    allow_llm_master_json: bool = Field(default=True, description="Usa LLM per generare master JSON da estratti")
    allow_llm_legal_analyzer: bool = Field(default=True, description="Usa LLM per analisi legale aggregata")
    legal_analyzer_model: str = Field(default="qwen72b:latest", description="Legal reasoning — Qwen2.5-14B-Instruct")
    synthesis_model: str = Field(default="qwen72b:latest", description="Modello LLM per relazione NPL finale. Primario: qwen72b:latest | Fallback: mistral-nemo:12b. Override via .env: SYNTHESIS_MODEL=")
    # PATCH 21 — sintesi finale via Claude Sonnet (Bedrock EU)
    synthesis_backend: str = Field(default="ollama", description="Backend sintesi: 'ollama' | 'bedrock' (Claude Sonnet EU) | 'bedrock_openai' (GPT-OSS 120B)")
    claude_model: str = Field(default="eu.anthropic.claude-sonnet-4-6", description="Modello Claude Bedrock EU cross-region inference")
    claude_bedrock_region: str = Field(default="eu-central-1", description="Regione AWS Bedrock. EU cross-region: eu-central-1. US: eu-central-1.")
    gpt_oss_model: str = Field(default="openai.gpt-oss-120b-1:0", description="GPT-OSS 120B via AWS Bedrock (Converse API). Nessuna attesa IAM.")
    gpt_oss_bedrock_region: str = Field(default="eu-central-1", description="Regione AWS per GPT-OSS Bedrock")
    gpt_oss_proxy: str = Field(default="", description="Proxy HTTP per Bedrock su SageMaker (es. http://proxy.example.internal:3128). Vuoto = nessun proxy.")
    bedrock_max_tokens: int = Field(default=32000, description="Max token output per chiamate Bedrock generiche (Qwen3, futuri modelli). Override via .env: BEDROCK_MAX_TOKENS=")
    qwen3_model: str = Field(default="qwen.qwen3-235b-a22b-2507-v1:0", description="Qwen3 235B via AWS Bedrock. Override via .env: QWEN3_MODEL=")
    bedrock_region: str = Field(default="eu-central-1", description="Regione AWS Bedrock per Qwen3. Override via .env: BEDROCK_REGION=")

    # === Multi-Agent Report (6 agenti Agno su Bedrock) ===
    multi_agent_mode: bool = Field(default=False, description="Attiva architettura multi-agente (6 agenti 235B) per la relazione NPL. Override via .env: MULTI_AGENT_MODE=true")

    # === Output Report (v4.0) ===
    # Disabilita i vecchi output PDF/XLS e usa solo il nuovo RELAZIONE.md
    enable_legacy_pdf_report: bool = Field(default=False, description="Genera il vecchio PDF con ReportGenerator (legacy). Default False: usa solo RELAZIONE.md")
    raw_json_mode: bool = Field(default=False, description="Bypass FactAssembler: invia tutti i JSON crudi (senza pre-processing Python) direttamente al LLM via prompt. Attiva con raw_json_mode: true in config.yml")
    enable_legacy_excel: bool = Field(default=True, description="Genera CHECKLIST.xlsx — output standard del bundle v4.1. Default True.")

    # === Review / Validation cycle (draft→validate→revise) ===
    review_enabled: bool = Field(default=False, description="Attiva ciclo draft→validate→revise per la relazione NPL finale")
    validation_temperature: float = Field(default=0.1, description="Temperature per il pass di validazione LLM")
    validation_max_tokens: int = Field(default=1500, description="Max token output per il pass di validazione LLM")
    validation_num_ctx: int = Field(default=32768, description="Context window (token) per il pass di validazione LLM")
    review_temperature: float = Field(default=0.1, description="Temperature per il pass di revisione LLM")
    review_max_tokens: int = Field(default=12000, description="Max token output per il pass di revisione LLM")
    review_num_ctx: int = Field(default=65536, description="Context window (token) per il pass di revisione LLM")

    ocr_cleanup_model: str = Field(default="qwen72b:latest", description="Fallback sintesi e pulizia OCR — Qwen2.5-14B-Instruct")
    
    # LLM context window e output — estrazione e sintesi SEQUENZIALI: ogni modello ha tutta la VRAM da solo
    # VRAM disponibile per Ollama: ~16.9GB totali (RTX)
    # deepseek-coder-v2:16b-lite (~9GB): estrazione JSON strutturato
    # mistral-nemo:12b (~7GB): sintesi relazione NPL (primario)
    # qwen72b:latest (~9GB): fallback sintesi / OCR cleanup
    ollama_json_num_ctx: int = Field(default=32768, description="Context window per chiamate JSON (entity extraction) — 32k tok con qwen3:8b")
    ollama_json_max_tokens: int = Field(default=5000, description="Max token output per chiamate JSON — 5000 tok supporta cernita 80 doc + entity extraction multi-garante")

    # Sintesi Map-Reduce (v4.0) — soglia e dimensione batch
    synthesis_map_batch_size: int = Field(default=10, description="Soglia n_docs oltre la quale attivare Map-Reduce nella sintesi (default 10)")
    synthesis_map_batch_size_per_batch: int = Field(default=12, description="Numero documenti per batch nel MAP step della sintesi (default 12 — testo OCR 4000 chars)")

    # Embedding model
    ollama_embed_model: str = Field(default="nomic-embed-text:latest", description="Modello per embeddings")
    ollama_base_url: str = Field(default="http://localhost:11434", description="URL Ollama server. Per llama_cpp.server usare http://localhost:11434/v1")
    llm_backend: str = Field(default="auto", description="Backend LLM: 'auto' (rileva da URL) | 'ollama' | 'openai' (llama_cpp.server)")
    ollama_request_timeout: float = Field(default=1800.0, description="Timeout richieste Ollama (10 min) - sufficiente per qwen3:8b / qwen3:30b-a3b su RTX")
    llm_doc_curation_timeout: float = Field(default=1800.0, description="Timeout dedicato Cernita LLM documenti (30 min) - batch da 80 doc richiedono più tempo del timeout standard")
    
    # === Paths ===
    data_input_path: Path = Field(default=Path("./data/input"))
    data_output_path: Path = Field(default=Path("./data/output"))
    data_processed_path: Path = Field(default=Path("./data/processed"))
    vectorstore_path: Path = Field(default=Path("./data/vectorstore"))
    
    # === Multi-Fascicolo ===
    fascicoli_base_path: Path = Field(default=Path("./data/fascicoli"), description="Path base per fascicoli multipli")
    
    # === OCR Configuration ===
    tesseract_cmd: str = Field(default=r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    poppler_path: str = Field(default=r"C:\Program Files\poppler-24.02.0\Library\bin")
    ocr_dpi: int = Field(default=300)
    ocr_language: str = Field(default="ita")
    enable_paddleocr: bool = Field(default=False, description="Usa PaddleOCR come fallback OCR. Default False: Surya → Tesseract. Override via .env: ENABLE_PADDLEOCR=true")
    enable_easyocr: bool = Field(default=False, description="Usa EasyOCR come primo OCR backend (può essere instabile su alcuni ambienti Torch)")
    paddleocr_use_gpu: bool = Field(default=True, description="Abilita GPU per PaddleOCR")
    surya_device: str = Field(default="cpu", description="Device PyTorch per Surya OCR (es. 'cuda', 'cuda:0', 'cpu'). Default cpu su SageMaker (driver mismatch cu130 vs CUDA 12.4)")
    paddleocr_lang: str = Field(default="it", description="Lingua PaddleOCR")
    ocr_max_pages: int = Field(default=0, description="Max pagine da processare con OCR (0 = nessun limite)")
    ocr_preprocess_contrast: bool = Field(default=True, description="Aumenta contrasto immagini prima di OCR")
    ocr_preprocess_grayscale: bool = Field(default=True, description="Converti in scala di grigi prima di OCR")
    ocr_cleanup_enabled: bool = Field(default=False, description="Abilita pulizia testo OCR con LLM")
    ocr_cleanup_confidence_threshold: float = Field(default=0.85, description="Soglia confidence OCR sopra cui saltare pulizia LLM (fast-path)")
    
    checkpoints_path: Path = Field(default=Path("./data/checkpoints"), description="Directory checkpoint ripresa elaborazione. Override via env: CHECKPOINTS_PATH")

    # === Database ===
    database_url: str = Field(default="sqlite+aiosqlite:///./data/npl_database.db")
    
    # === Processing - RTX 4060 Ti 16GB OPTIMIZED ===
    max_workers: int = Field(default=12, description="Worker paralleli - bilanciato con LLM concurrent")
    batch_size: int = Field(default=32, description="Batch size per GPU inference")
    enable_gpu_optimization: bool = Field(default=True, description="Attiva ottimizzazioni GPU")
    llm_max_concurrent: int = Field(default=4, description="Limite chiamate LLM concorrenti - aumentato per 16GB VRAM")

    # === GPU Memory Management ===
    vram_keep_models_threshold_mb: int = Field(
        default=12288,
        description="Soglia VRAM (MB) sotto cui NON scaricare i modelli (default: 12GB)"
    )

    # === Prefetch & Cache ===
    ocr_prefetch_look_ahead: int = Field(default=4, description="Numero documenti prefetch in coda (VRAM 16GB)")
    ocr_prefetch_queue_size: int = Field(default=8, description="Dimensione coda prefetch OCR")
    ocr_cache_max_size: int = Field(default=200, description="Dimensione cache OCR per workflow paralleli")
    llm_cache_dir: Path = Field(default=Path("./data/cache/llm"), description="Cache persistente LLM")
    llm_cache_ttl_hours: float = Field(default=168.0, description="TTL cache LLM in ore")

    # === PDF Parsing ===
    enable_pymupdf4llm: bool = Field(default=False, description="Usa PyMuPDF4LLM per PDF")
    pdf_parse_timeout: float = Field(default=30.0, description="Timeout parsing PDF (secondi)")
    enable_pymupdf_layout: bool = Field(default=False, description="Usa pymupdf_layout per layout PDF")
    enable_pdfplumber_layout: bool = Field(default=True, description="Usa pdfplumber per layout PDF")
    
    # === Smart Pipeline (Florence-2 + Routing Intelligente) ===
    enable_smart_pipeline: bool = Field(default=True, description="Abilita pipeline intelligente con Florence-2")
    enable_florence2: bool = Field(default=False, description="Usa Florence-2 per analisi visiva (abilitare solo su stack torch stabile)")
    enable_opencv_preprocessing: bool = Field(default=True, description="Preprocessing OpenCV per scansioni bassa qualità")
    signature_detection_threshold: float = Field(default=0.65, description="Soglia confidence rilevamento firme")
    quality_preprocessing_threshold: float = Field(default=0.6, description="Sotto questa qualità applica preprocessing")
    
    # === Logging ===
    log_level: str = Field(default="INFO")
    log_file: Optional[Path] = Field(default=Path("./data/logs/npl_agent.log"))

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Singleton per accesso globale
settings = Settings()



# Crea directory necessarie all'avvio
def ensure_directories():
    """Crea le directory necessarie se non esistono"""
    dirs = [
        settings.data_input_path,
        settings.data_output_path,
        settings.data_processed_path,
        settings.vectorstore_path,
        settings.fascicoli_base_path,
        settings.checkpoints_path,
        Path("./data/logs"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
