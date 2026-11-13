# -*- coding: utf-8 -*-
# © 2026 Gioacchino Tortorici — ZT Legal Analysis NPL Intelligence Platform
# Licenza: PolyForm Small Business 1.0.0 (vedi LICENSE.md).
"""
PersistentCache - Cache persistente per risposte LLM

Sostituisce llm_cache.py e cached_llm_client.py con un'unica implementazione.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Any

from loguru import logger


@dataclass
class CacheEntry:
    """Entry nella cache persistente"""
    content: str
    model: str
    tokens_used: int
    timestamp: float
    cache_key: str
    hit_count: int = 0


@dataclass
class CacheStats:
    """Statistiche cache"""
    hits: int = 0
    misses: int = 0
    total_tokens_saved: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PersistentCache:
    """
    Cache persistente su disco basata su hash (prompt+system+model).
    """

    def __init__(
        self,
        cache_dir: Path,
        ttl_hours: float = 168.0,
        max_entries: int = 2000,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.max_entries = max_entries

        self._lock = Lock()
        self._stats = CacheStats()
        self._index: Dict[str, str] = {}

        self._load_index()
        logger.info(f"💾 PersistentCache initialized: {len(self._index)} entries, TTL={ttl_hours}h")

    def _make_key(self, prompt: str, system_prompt: Optional[str], model: str) -> str:
        content = f"{model}::{system_prompt or ''}::{prompt}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _load_index(self):
        index_path = self.cache_dir / "_index.json"
        if index_path.exists():
            try:
                self._index = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"⚠️ Cache index load failed: {e}")
                self._index = {}

    def _save_index(self):
        index_path = self.cache_dir / "_index.json"
        try:
            index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"⚠️ Cache index save failed: {e}")

    def get(self, prompt: str, system_prompt: Optional[str], model: str) -> Optional[CacheEntry]:
        key = self._make_key(prompt, system_prompt, model)
        with self._lock:
            if key not in self._index:
                self._stats.misses += 1
                return None

            cache_file = self.cache_dir / f"{key}.json"
            if not cache_file.exists():
                self._index.pop(key, None)
                self._stats.misses += 1
                return None

            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                entry = CacheEntry(**data)

                if time.time() - entry.timestamp > self.ttl_seconds:
                    cache_file.unlink(missing_ok=True)
                    self._index.pop(key, None)
                    self._stats.misses += 1
                    return None

                entry.hit_count += 1
                self._stats.hits += 1
                self._stats.total_tokens_saved += entry.tokens_used

                cache_file.write_text(json.dumps(asdict(entry), indent=2), encoding="utf-8")
                return entry
            except Exception as e:
                logger.warning(f"⚠️ Cache read failed for {key[:8]}: {e}")
                self._stats.misses += 1
                return None

    def set(self, prompt: str, content: str, model: str, tokens_used: int = 0, system_prompt: Optional[str] = None):
        key = self._make_key(prompt, system_prompt, model)
        entry = CacheEntry(
            content=content,
            model=model,
            tokens_used=tokens_used,
            timestamp=time.time(),
            cache_key=key,
        )

        with self._lock:
            if len(self._index) >= self.max_entries:
                self._evict_oldest()

            cache_file = self.cache_dir / f"{key}.json"
            try:
                cache_file.write_text(json.dumps(asdict(entry), indent=2), encoding="utf-8")
                self._index[key] = cache_file.name
                self._save_index()
            except Exception as e:
                logger.warning(f"⚠️ Cache write failed for {key[:8]}: {e}")

    def _evict_oldest(self):
        if not self._index:
            return

        oldest_key = None
        oldest_time = float("inf")
        for cache_key in list(self._index.keys())[:200]:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    if data.get("timestamp", 0) < oldest_time:
                        oldest_time = data.get("timestamp", 0)
                        oldest_key = cache_key
                except Exception:
                    continue

        if oldest_key:
            cache_file = self.cache_dir / f"{oldest_key}.json"
            cache_file.unlink(missing_ok=True)
            self._index.pop(oldest_key, None)

    def clear(self):
        with self._lock:
            for key in list(self._index.keys()):
                cache_file = self.cache_dir / f"{key}.json"
                cache_file.unlink(missing_ok=True)
            self._index.clear()
            self._save_index()
            self._stats = CacheStats()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "entries": len(self._index),
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "hit_rate": f"{self._stats.hit_rate:.1%}",
            "tokens_saved": self._stats.total_tokens_saved,
            "cache_dir": str(self.cache_dir),
        }
