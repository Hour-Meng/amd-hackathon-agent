"""Tier 0 — FAISS semantic cache gate for zero-token repeat queries."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from my_routing_agent.config import CacheConfig

logger = logging.getLogger("semantic_cache")

try:
    import faiss
    from sentence_transformers import SentenceTransformer

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    SentenceTransformer = None


@dataclass
class CacheEntry:
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    hits: int = 0


class SemanticCache:
    """FAISS + MiniLM semantic cache for intercepting semantically identical queries."""

    def __init__(self, config: CacheConfig | None = None) -> None:
        self._config = config or CacheConfig()
        self._encoder: Any = None
        self._index: Any = None
        self._store: dict[int, CacheEntry] = {}
        self._next_id: int = 0
        self._dimension: int = 384
        self._initialized: bool = False

    def initialize(self) -> bool:
        if not FAISS_AVAILABLE:
            logger.warning("faiss-cpu or sentence-transformers not installed; cache disabled")
            return False
        try:
            model_name = self._config.model_name
            logger.info("Loading encoder: %s", model_name)
            self._encoder = SentenceTransformer(model_name)
            self._dimension = self._encoder.get_sentence_embedding_dimension()
            self._index = faiss.IndexFlatIP(self._dimension)
            self._initialized = True
            self._load_persisted()
            logger.info("Semantic cache initialized (dim=%d, entries=%d)", self._dimension, len(self._store))
            return True
        except Exception as exc:
            logger.warning("Cache init failed: %s", exc)
            return False

    def lookup(self, query: str) -> CacheEntry | None:
        if not self._initialized or not query.strip():
            return None
        embedding = self._encode(query)
        if embedding is None:
            return None
        if self._index.ntotal == 0:
            return None
        try:
            similarities, indices = self._index.search(embedding.reshape(1, -1), k=1)
            similarity = float(similarities[0][0])
            idx = int(indices[0][0])
            if similarity >= self._config.threshold and idx in self._store:
                entry = self._store[idx]
                entry.hits += 1
                logger.info("CACHE HIT sim=%.4f idx=%d hits=%d", similarity, idx, entry.hits)
                return entry
        except Exception as exc:
            logger.warning("Cache lookup error: %s", exc)
        return None

    def store(self, query: str, response: str, metadata: dict[str, Any] | None = None) -> None:
        if not self._initialized or not query.strip() or not response.strip():
            return
        embedding = self._encode(query)
        if embedding is None:
            return
        idx = self._next_id
        self._next_id += 1
        self._index.add(embedding.reshape(1, -1))
        self._store[idx] = CacheEntry(
            response=response,
            metadata=metadata or {},
            timestamp=time.time(),
        )
        logger.info("CACHE STORE idx=%d response_len=%d", idx, len(response))

    def clear(self) -> None:
        if not self._initialized:
            return
        self._index.reset()
        self._store.clear()
        self._next_id = 0
        logger.info("Cache cleared")

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self._index.ntotal
        if total == 0:
            return 0.0
        total_hits = sum(e.hits for e in self._store.values())
        return total_hits / (total + total_hits) if (total + total_hits) > 0 else 0.0

    def save_persisted(self) -> None:
        if not self._initialized or not self._store:
            return
        try:
            faiss.write_index(self._index, self._config.index_path)
            serializable = {
                str(k): {
                    "response": v.response,
                    "metadata": v.metadata,
                    "timestamp": v.timestamp,
                    "hits": v.hits,
                }
                for k, v in self._store.items()
            }
            with open(self._config.store_path, "w") as f:
                json.dump(serializable, f)
            logger.info("Cache persisted (%d entries)", len(self._store))
        except Exception as exc:
            logger.warning("Cache persist error: %s", exc)

    def _load_persisted(self) -> None:
        idx_path = Path(self._config.index_path)
        store_path = Path(self._config.store_path)
        if idx_path.exists() and store_path.exists():
            try:
                self._index = faiss.read_index(str(idx_path))
                with open(store_path) as f:
                    raw = json.load(f)
                max_key = 0
                for k, v in raw.items():
                    kid = int(k)
                    self._store[kid] = CacheEntry(
                        response=v["response"],
                        metadata=v.get("metadata", {}),
                        timestamp=v.get("timestamp", 0.0),
                        hits=v.get("hits", 0),
                    )
                    max_key = max(max_key, kid)
                self._next_id = max_key + 1
                logger.info("Loaded %d cached entries from disk", len(self._store))
            except Exception as exc:
                logger.warning("Cache load error (starting fresh): %s", exc)
                self._index = faiss.IndexFlatIP(self._dimension)
                self._store = {}

    def _encode(self, text: str) -> np.ndarray | None:
        try:
            return self._encoder.encode(text, normalize_embeddings=True)
        except Exception as exc:
            logger.warning("Encoding error: %s", exc)
            return None
