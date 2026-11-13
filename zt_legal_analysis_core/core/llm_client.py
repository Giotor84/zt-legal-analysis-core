# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
LLM Client - Interfaccia unificata per Ollama
Multi-LLM Strategy: Qwen2.5-Coder | Gemma2 | Llama-3.1
"""

import ollama
import random
from typing import Dict, List, Optional, Generator, Any
from dataclasses import dataclass
from loguru import logger
from threading import Lock

from zt_legal_analysis_core.core.config import settings
from zt_legal_analysis_core.infrastructure.cache.persistent_cache import PersistentCache


@dataclass
class LLMResponse:
    """Risposta dal modello LLM"""
    content: str
    model: str
    tokens_used: int = 0
    

class OllamaClient:
    """
    Client per interazione con Ollama.
    Supporta modelli multipli specializzati per task diversi.
    """
    
    _WARMUP_LOCK = Lock()
    _WARMED_MODELS = set()

    def __init__(self, model: Optional[str] = None, cache: Optional[PersistentCache] = None, timeout: Optional[float] = None):
        self.model = model or settings.ollama_model
        self.base_url = settings.ollama_base_url
        self.request_timeout = timeout if timeout is not None else getattr(settings, "ollama_request_timeout", 1800.0)
        self.cache = cache
        # Rileva automaticamente se il backend è OpenAI-compatibile (llama_cpp.server)
        # oppure Ollama nativo. Attivato se base_url termina con /v1 o contiene /v1/
        _backend = str(getattr(settings, "llm_backend", "auto")).lower()
        if _backend == "openai":
            self._openai_api = True
        elif _backend == "ollama":
            self._openai_api = False
        else:  # auto: rileva dalla URL
            self._openai_api = "/v1" in self.base_url
        self._client = self._init_client()
        self._verify_connection()
        self._warmup_gpu()

    def _init_client(self):
        """Inizializza il client Ollama con timeout, se supportato."""
        if self._openai_api:
            return None  # llama_cpp.server: usiamo requests direttamente
        client_cls = getattr(ollama, "Client", None)
        if not client_cls:
            return None

        try:
            return client_cls(host=self.base_url, timeout=self.request_timeout)
        except TypeError:
            # Versioni SDK vecchie non accettano il kwarg timeout.
            # Tentiamo monkey-patch del client httpx interno per preservare il timeout.
            try:
                import httpx
                client = client_cls(host=self.base_url)
                ht = httpx.Timeout(self.request_timeout, connect=30.0)
                for attr in ("_client", "session", "_session"):
                    inner = getattr(client, attr, None)
                    if isinstance(inner, httpx.Client):
                        inner.timeout = ht
                        logger.debug(f"⏱️ Timeout Ollama impostato via patch httpx: {self.request_timeout}s")
                        break
                return client
            except Exception as e:
                logger.warning(f"⚠️ Impossibile impostare timeout su ollama.Client: {e}")
                try:
                    return client_cls(host=self.base_url)
                except Exception as e2:
                    logger.warning(f"⚠️ Impossibile inizializzare ollama.Client: {e2}")
                    return None

    def _list_models(self) -> List[str]:
        """Lista modelli disponibili (robusto a formati diversi di risposta)."""
        if self._openai_api:
            import requests as _rq
            r = _rq.get(f"{self.base_url.rstrip('/')}/v1/models", timeout=10)
            if r.status_code == 200:
                data = r.json().get("data", [])
                return [m.get("id", "") for m in data if isinstance(m, dict)]
            return [self.model]  # fallback: assume il modello sia disponibile
        # Timeout nativo gestito da ollama.Client(timeout=self.request_timeout)
        response = (self._client.list if self._client else ollama.list)()

        models_raw: List[Any] = []
        if isinstance(response, dict):
            models_raw = response.get("models") or response.get("data") or []
        elif isinstance(response, list):
            models_raw = response

        model_names: List[str] = []
        for m in models_raw:
            if isinstance(m, str):
                model_names.append(m)
                continue
            if isinstance(m, dict):
                name = m.get("name") or m.get("model") or m.get("id")
                if name:
                    model_names.append(name)

        return model_names
    
    def _verify_connection(self) -> bool:
        """Verifica che il backend LLM sia attivo e il modello disponibile"""
        try:
            if self._openai_api:
                # llama_cpp.server: verifica con /health o /v1/models
                import requests as _rq
                base = self.base_url.rstrip('/')
                # Rimuovi /v1 finale se presente per costruire correttamente l'URL
                base_root = base[:-3] if base.endswith('/v1') else base
                r = _rq.get(f"{base_root}/health", timeout=5)
                ok = r.status_code == 200
                if ok:
                    logger.info(f"✅ llama_cpp.server connesso - Modello: {self.model}")
                else:
                    logger.warning(f"⚠️ llama_cpp.server /health → {r.status_code}")
                return ok

            model_names = self._list_models()
            if not model_names:
                return False
            
            if self.model not in model_names:
                # Prova senza tag :latest
                base_model = self.model.split(':')[0]
                available = [m for m in model_names if m.startswith(base_model)]
                if available:
                    self.model = available[0]
                    logger.info(f"Usando modello: {self.model}")
                else:
                    logger.warning(f"Modello {self.model} non trovato. Disponibili: {model_names}")
                    return False
            
            logger.info(f"✅ Ollama connesso - Modello: {self.model}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Errore connessione LLM backend: {e}")
            return False
    
    def _warmup_gpu(self) -> None:
        """Precarica il modello in GPU VRAM per eliminare latency"""
        warmup_enabled = bool(getattr(settings, "ollama_warmup_enabled", False))
        if not warmup_enabled:
            return

        warmup_once = bool(getattr(settings, "ollama_warmup_once_per_model", True))
        if warmup_once:
            with self._WARMUP_LOCK:
                if self.model in self._WARMED_MODELS:
                    return

        try:
            logger.info("🔥 Preloading modello in GPU VRAM...")
            client = self._client or ollama
            client.chat(
                model=self.model,
                messages=[{"role": "user", "content": "warmup"}],
                options={"num_predict": 1}
            )
            logger.info("✅ Modello caricato in VRAM - latency eliminata")
            if warmup_once:
                with self._WARMUP_LOCK:
                    self._WARMED_MODELS.add(self.model)
        except Exception as e:
            logger.warning(f"⚠️ GPU warmup fallito (non critico): {e}")
    
    def chat(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        use_cache: bool = True,
        response_format: Optional[str] = None,
        num_ctx: Optional[int] = None,
        think: Optional[bool] = None,
        keep_alive: Optional[str] = None,
    ) -> LLMResponse:
        """
        Invia una richiesta chat al modello.
        
        Args:
            prompt: Il prompt dell'utente
            system_prompt: Istruzioni di sistema opzionali
            temperature: Creatività (0.0 = deterministico, 1.0 = creativo)
            max_tokens: Massimo numero di token in risposta
            use_cache: Se True, usa cache per risposte identiche
        
        Returns:
            LLMResponse con il contenuto della risposta
        """
        # Check cache se abilitata e temperature bassa (deterministico)
        if use_cache and temperature <= 0.1 and self.cache:
            cached = self.cache.get(prompt, system_prompt, self.model)
            if cached:
                return LLMResponse(
                    content=cached.content,
                    model=cached.model,
                    tokens_used=cached.tokens_used
                )
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            client = self._client or ollama
            
            temp = max(float(temperature), 0.0)
            llm_options = {
                "temperature": temp,
                "num_predict": max_tokens,
                "num_ctx": num_ctx or 8192,  # Caller può richiedere contesto più grande (es. gemma3 sintesi a 65536)
                "num_gpu": 999,              # Forza tutti i layer su GPU (Ollama cap automatico se VRAM insufficiente)
                "num_batch": 2048,           # Prompt eval più veloce (default Ollama: 512) — riduce il tempo
                                             # di "ingestion" dei JSON estratti prima della generazione
            }

            if response_format == "json":
                # Se num_ctx esplicito, rispettalo; altrimenti usa settings
                if not num_ctx:
                    llm_options["num_ctx"] = int(getattr(settings, "ollama_json_num_ctx", 8192))
                # Rispetta max_tokens esplicito; usa cap settings solo come fallback
                json_default_cap = int(getattr(settings, "ollama_json_max_tokens", 3000))
                if max_tokens <= 2000:  # caller non ha specificato valori alti
                    llm_options["num_predict"] = min(max_tokens, json_default_cap)
                # else: keep the caller's explicit max_tokens

            if temp > 0:
                llm_options["seed"] = random.randint(1, 999999)
            
            # Se temperature è bassa (estrazione dati), aggiungi repeat_penalty
            if temperature <= 0.1:
                llm_options["repeat_penalty"] = 1.1  # Evita ripetizioni
            
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "options": llm_options,
            }
            if response_format:
                request_kwargs["format"] = response_format
            # think=False disabilita il blocco <think> dei modelli reasoning (Ollama 0.16+)
            # Risparmia 2000-3000 token di ragionamento → tutti i token vanno all'output
            if think is not None:
                request_kwargs["think"] = think
            # keep_alive: mantiene il modello in VRAM tra batch lunghi (default Ollama: 5 min)
            # Forzato a 2h se non specificato dal caller — evita reload durante batch OCR lunghi
            request_kwargs["keep_alive"] = keep_alive if keep_alive is not None else "2h"

            import requests as _rq_llm
            if self._openai_api:
                # llama_cpp.server — OpenAI /v1/chat/completions
                base = self.base_url.rstrip('/')
                base_root = base[:-3] if base.endswith('/v1') else base
                _chat_url = f"{base_root}/v1/chat/completions"
                _oai_payload = {
                    "model": request_kwargs["model"],
                    "messages": request_kwargs["messages"],
                    "max_tokens": llm_options.get("num_predict", max_tokens),
                    "temperature": llm_options.get("temperature", temperature),
                    "stream": False,
                }
                if response_format == "json":
                    _oai_payload["response_format"] = {"type": "json_object"}
                _chat_r = _rq_llm.post(_chat_url, json=_oai_payload, timeout=self.request_timeout)
                if _chat_r.status_code == 200:
                    _resp_json = _chat_r.json()
                    content = _resp_json["choices"][0]["message"]["content"]
                    tokens_used = (_resp_json.get("usage") or {}).get("completion_tokens", 0)
                else:
                    raise RuntimeError(f"llama_cpp.server HTTP {_chat_r.status_code}: {_chat_r.text[:300]}")
            else:
                # Ollama nativo — /api/chat
                _chat_url = f"{self.base_url.rstrip('/')}/api/chat"
                _chat_payload = dict(request_kwargs)
                _chat_payload["stream"] = False
                _chat_r = _rq_llm.post(_chat_url, json=_chat_payload, timeout=self.request_timeout)
                if _chat_r.status_code == 200:
                    response = _chat_r.json()
                else:
                    raise RuntimeError(f"Ollama HTTP {_chat_r.status_code}: {_chat_r.text[:300]}")

                if response is None:
                    raise TimeoutError("Risposta Ollama vuota o timeout")

                if isinstance(response, dict):
                    content = response['message']['content']
                    tokens_used = response.get('eval_count', 0)
                else:
                    content = response.message.content
                    tokens_used = getattr(response, 'eval_count', 0) or 0

            result = LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used
            )
            
            # Salva in cache se abilitata
            if use_cache and temperature <= 0.1 and self.cache:
                self.cache.set(
                    prompt=prompt,
                    content=result.content,
                    model=self.model,
                    tokens_used=result.tokens_used,
                    system_prompt=system_prompt
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Errore chat Ollama: {e}")
            raise
    
    def chat_stream(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1
    ) -> Generator[str, None, None]:
        """
        Chat con streaming per feedback real-time.
        
        Yields:
            Chunks di testo man mano che vengono generati
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            client = self._client or ollama
            stream = client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={"temperature": temperature}
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            logger.error(f"Errore streaming Ollama: {e}")
            raise
    
    def generate_json(
        self, 
        prompt: str, 
        schema_hint: str = "",
        temperature: float = 0.0
    ) -> Dict:
        """
        Genera output JSON strutturato.
        
        Args:
            prompt: Prompt con richiesta
            schema_hint: Esempio dello schema JSON atteso
            temperature: Usa 0.0 per output deterministico
        
        Returns:
            Dizionario parsato dalla risposta JSON
        """
        import json
        import re
        
        full_prompt = f"""{prompt}

Rispondi ESCLUSIVAMENTE con un JSON valido, senza altro testo.
{f"Schema atteso: {schema_hint}" if schema_hint else ""}"""
        
        response = self.chat(prompt=full_prompt, temperature=temperature)
        
        # Estrai JSON dalla risposta
        content = response.content
        
        # Prova a trovare JSON nel testo
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Prova array JSON
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        logger.warning(f"Impossibile parsare JSON dalla risposta: {content[:200]}")
        return {"raw_response": content, "parse_error": True}


# ---------------------------------------------------------------------------
# BedrockClient — Amazon Bedrock Converse API
# Modelli supportati: openai.gpt-oss-120b-1:0, amazon.nova-pro-v1:0, ecc.
# Credenziali: variabili d'ambiente AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
#              oppure IAM Role SageMaker (automatico su istanze ml.*)
# ---------------------------------------------------------------------------

class BedrockClient:
    """Client per Amazon Bedrock via Converse API.

    Interfaccia identica a OllamaClient.chat() — drop-in replacement.
    Parametri Ollama-specifici (num_ctx, keep_alive, think) vengono ignorati.
    """

    BEDROCK_PREFIXES = ("openai.", "amazon.", "anthropic.", "meta.", "mistral.", "cohere.", "qwen.", "minimax.", "nvidia.")

    def __init__(self, model: str, region: str = "eu-central-1", timeout: float = 1800.0, proxy: str = ""):
        import boto3
        from botocore.config import Config

        self.model = model
        self.region = region
        self.request_timeout = timeout

        # Proxy HTTP (obbligatorio in ambiente SageMaker Studio con proxy aziendale)
        _proxy = proxy or str(getattr(settings, "gpt_oss_proxy", ""))
        _proxies = {"http": _proxy, "https": _proxy} if _proxy else None

        _cfg_kwargs: dict = {
            "read_timeout": int(timeout),
            "connect_timeout": 30,
            "retries": {"max_attempts": 2, "mode": "standard"},
        }
        if _proxies:
            _cfg_kwargs["proxies"] = _proxies

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(**_cfg_kwargs),
        )
        logger.info(f"✅ BedrockClient inizializzato — modello: {model}, region: {region}" + (f", proxy: {_proxy}" if _proxy else ""))

    @classmethod
    def is_bedrock_model(cls, model_id: str) -> bool:
        return any(model_id.startswith(p) for p in cls.BEDROCK_PREFIXES)

    def _preview(self, value: Any, max_len: int = 1200) -> str:
        """Anteprima sicura per i log."""
        try:
            text = str(value)
        except Exception:
            text = repr(value)
        text = text.replace("\n", "\\n")
        if len(text) > max_len:
            return text[:max_len] + "... [TRUNCATED]"
        return text

    def _extract_text_from_content_blocks(self, content_blocks: Any) -> Optional[str]:
        """Estrae tutto il testo utile da Bedrock content blocks.

        Gestisce blocchi tipo:
        - {"text": "..."}
        - {"reasoningContent": {...}}  -> ignorato
        - {"toolUse": {...}}           -> ignorato
        - {"toolResult": {...}}        -> ignorato
        """
        if content_blocks is None:
            return None

        # Se per qualche motivo è già una stringa, restituiscila direttamente
        if isinstance(content_blocks, str):
            text = content_blocks.strip()
            return text if text else None

        if not isinstance(content_blocks, list):
            return None

        collected: List[str] = []

        for idx, block in enumerate(content_blocks):
            if not isinstance(block, dict):
                logger.debug(f"[Bedrock.chat] content[{idx}] non è dict: {type(block).__name__}")
                continue

            # Caso standard Bedrock Converse: blocco testuale
            if "text" in block and isinstance(block["text"], str):
                txt = block["text"].strip()
                if txt:
                    collected.append(txt)
                continue

            # Alcuni provider possono usare content annidato
            # Esempio prudenziale: {"text": {"value": "..."}}
            if "text" in block and isinstance(block["text"], dict):
                nested = block["text"].get("value")
                if isinstance(nested, str) and nested.strip():
                    collected.append(nested.strip())
                continue

            # Reasoning/tool blocks: log di debug ma non errore
            logger.debug(
                f"[Bedrock.chat] content[{idx}] keys={list(block.keys())} "
                f"(blocco non testuale ignorato)"
            )

        if collected:
            return "\n".join(collected).strip()

        return None

    def _extract_response_text(self, response: Dict[str, Any]) -> str:
        """Estrae il testo dalla risposta Bedrock in modo robusto.

        Supporta:
        - Bedrock Converse API standard
        - eventuali wrapper tipo OpenAI-style
        - fallback su strutture alternative
        """
        if not isinstance(response, dict):
            raise ValueError(f"Risposta Bedrock non valida: type={type(response).__name__}")

        # ------------------------------------------------------------------
        # 1) Bedrock Converse API standard
        # ------------------------------------------------------------------
        output = response.get("output")
        if isinstance(output, dict):
            message = output.get("message")
            if isinstance(message, dict):
                content_blocks = message.get("content")
                text = self._extract_text_from_content_blocks(content_blocks)
                if text:
                    return text

        # ------------------------------------------------------------------
        # 2) Fallback: OpenAI-style wrapper
        # ------------------------------------------------------------------
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                    text = self._extract_text_from_content_blocks(content)
                    if text:
                        return text

        # ------------------------------------------------------------------
        # 3) Fallback: anthropic-like / altri wrapper
        # ------------------------------------------------------------------
        content = response.get("content")
        text = self._extract_text_from_content_blocks(content)
        if text:
            return text

        # ------------------------------------------------------------------
        # 4) Fallback: generation semplice
        # ------------------------------------------------------------------
        generation = response.get("generation")
        if isinstance(generation, str) and generation.strip():
            return generation.strip()

        # ------------------------------------------------------------------
        # Niente testo trovato -> errore descrittivo
        # ------------------------------------------------------------------
        summary = {
            "top_level_keys": list(response.keys()),
            "output_keys": list(output.keys()) if isinstance(output, dict) else None,
            "message_keys": list(message.keys()) if isinstance(output, dict) and isinstance(message, dict) else None,
        }
        raise ValueError(f"Nessun blocco testuale trovato nella risposta Bedrock: {summary}")

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 12000,
        # Parametri Ollama ignorati su Bedrock:
        use_cache: bool = False,
        response_format: Optional[str] = None,
        num_ctx: Optional[int] = None,
        think: Optional[bool] = None,
        keep_alive: Optional[str] = None,
    ) -> LLMResponse:
        """Chiama Bedrock via Converse API.

        Parametri Ollama-specifici ignorati:
        - use_cache
        - response_format
        - num_ctx
        - keep_alive

        Parametro attivo su Bedrock:
        - think=True  → Qwen3 Extended Thinking (additionalModelRequestFields, budget=max_tokens//2)
        - think=False → Qwen3 fast-mode (thinking disabilitato)
        - think=None  → comportamento default del modello
        """
        # Preemptive context cap: stima grossolana (primo filtro).
        # Il contenuto misto JSON+italiano tokenizza a ~2.3-2.5 chars/token per Qwen3.
        # NOTA: il campo 'value=N' nel ValidationException di Bedrock è = 262145-max_tokens,
        # che coincide con i token reali di input (total=ctx+1 → input=ctx+1-output).
        # Il retry intelligente sotto usa questo valore per una seconda chiamata esatta.
        _BEDROCK_CTX_LIMIT = 262144
        _BEDROCK_SAFETY_MARGIN = 8192  # copre variabilità ratio chars/token (~3-8% errore)
        _est_input_tok = int((len(prompt) + len(system_prompt or "")) / 2.5)
        if _est_input_tok + max_tokens > _BEDROCK_CTX_LIMIT - _BEDROCK_SAFETY_MARGIN:
            _capped_max = max(4096, _BEDROCK_CTX_LIMIT - _BEDROCK_SAFETY_MARGIN - _est_input_tok)
            logger.warning(
                f"[Bedrock.chat] Preemptive cap: est_input={_est_input_tok} tok, "
                f"max_tokens {max_tokens} → {_capped_max}"
            )
            max_tokens = _capped_max

        request: Dict[str, Any] = {
            "modelId": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ],
            "inferenceConfig": {
                "maxTokens": int(max_tokens),
                "temperature": max(float(temperature), 0.0),
            },
        }

        if system_prompt:
            request["system"] = [{"text": system_prompt}]

        # Qwen3 su Bedrock: attiva thinking (Extended Thinking) via additionalModelRequestFields.
        # think=True → reasoning CoT; think=False/None → fast-mode (/no_think).
        # Budget: metà dei max_tokens disponibili, min 1024, max 20000.
        # Quando thinking è attivo, temperature DEVE essere 1.0 (requisito Qwen3 Bedrock).
        if "qwen" in self.model.lower() and think is True:
            _budget = min(max(int(max_tokens) // 2, 1024), 32000)
            request["additionalModelRequestFields"] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": _budget,
                }
            }
            # Temperature fissa a 1.0 quando thinking è abilitato (requisito Bedrock Qwen3)
            request["inferenceConfig"]["temperature"] = 1.0
            logger.info(
                f"[Bedrock] Qwen3 Extended Thinking ATTIVO — "
                f"budget_tokens={_budget}, max_tokens={max_tokens}"
            )
        elif "qwen" in self.model.lower() and think is False:
            # Disabilita esplicitamente il thinking
            request["additionalModelRequestFields"] = {
                "thinking": {"type": "disabled"}
            }
            logger.debug("[Bedrock] Qwen3 thinking DISABILITATO")

        # ------------------------------------------------------------------
        # DEBUG INPUT
        # ------------------------------------------------------------------
        logger.debug(
            f"[Bedrock.chat] START model={self.model} "
            f"prompt_chars={len(prompt or '')} "
            f"system_chars={len(system_prompt or '')}"
        )
        logger.debug(f"[Bedrock.chat] USER PREVIEW: {self._preview(prompt)}")
        if system_prompt:
            logger.debug(f"[Bedrock.chat] SYSTEM PREVIEW: {self._preview(system_prompt)}")

        # Se vuoi vedere TUTTO il payload, scommenta:
        # import json
        # logger.debug(f"[Bedrock.chat] REQUEST JSON: {json.dumps(request, ensure_ascii=False, default=str)}")

        try:
            # ------------------------------------------------------------------
            # CHIAMATA AWS (con retry automatico su overflow context window)
            # ------------------------------------------------------------------
            try:
                response = self._client.converse(**request)
            except Exception as _overflow_exc:
                import re as _re
                _m = _re.search(r'at least (\d+) input tokens', str(_overflow_exc))
                if _m:
                    _actual_input = int(_m.group(1))
                    _retry_max = max(4096, _BEDROCK_CTX_LIMIT - _actual_input - 1)
                    logger.warning(
                        f"[Bedrock.chat] Context overflow: input_reale={_actual_input} tok → "
                        f"retry con max_tokens={_retry_max}"
                    )
                    request["inferenceConfig"]["maxTokens"] = _retry_max
                    response = self._client.converse(**request)  # se fallisce propaga all'except esterno
                else:
                    raise

            logger.debug(
                f"[Bedrock.chat] RAW RESPONSE top-level keys: "
                f"{list(response.keys()) if isinstance(response, dict) else type(response).__name__}"
            )

            if isinstance(response, dict):
                output = response.get("output")
                if isinstance(output, dict):
                    message = output.get("message")
                    if isinstance(message, dict):
                        content_blocks = message.get("content")
                        if isinstance(content_blocks, list):
                            logger.debug(
                                f"[Bedrock.chat] content block count={len(content_blocks)} "
                                f"block_keys={[list(b.keys()) if isinstance(b, dict) else type(b).__name__ for b in content_blocks]}"
                            )

            # Se vuoi vedere TUTTA la risposta raw, scommenta:
            # import json
            # logger.debug(f"[Bedrock.chat] RAW RESPONSE JSON: {json.dumps(response, ensure_ascii=False, default=str)}")

            # ------------------------------------------------------------------
            # PARSING TESTO
            # ------------------------------------------------------------------
            content = self._extract_response_text(response)

            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            input_tokens = int(usage.get("inputTokens", 0) or 0)
            output_tokens = int(usage.get("outputTokens", 0) or 0)
            tokens_used = input_tokens + output_tokens

            logger.debug(
                f"[Bedrock.chat] OK input_tokens={input_tokens} "
                f"output_tokens={output_tokens} "
                f"content_chars={len(content or '')}"
            )
            logger.debug(f"[Bedrock.chat] OUTPUT PREVIEW: {self._preview(content)}")

            return LLMResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
            )

        except Exception as e:
            logger.exception(f"Errore BedrockClient.chat: {e}")
            raise


# Singleton per accesso globale
_client: Optional[OllamaClient] = None

def get_llm_client() -> OllamaClient:
    """Ottieni l'istanza singleton del client LLM"""
    global _client
    if _client is None:
        cache = PersistentCache(
            cache_dir=settings.llm_cache_dir,
            ttl_hours=settings.llm_cache_ttl_hours,
        )
        _client = OllamaClient(cache=cache)
    return _client
