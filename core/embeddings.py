"""
Phase 1 – Multilingual Embedding Engine
========================================
Provides:
  • SentenceTransformerEmbedder  – production wrapper (requires sentence-transformers)
  • TFIDFEmbedder                – pure-sklearn fallback for offline/test environments
  • EmbeddingEngine              – unified interface that selects the best available backend

All embedders share the same API:
    engine.embed_texts(texts)   → np.ndarray  shape (N, dim)
    engine.embed_query(query)   → np.ndarray  shape (1, dim)
"""

import logging
import math
import hashlib
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.decomposition import TruncatedSVD

from config import CONFIG, CACHE_DIR

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Base class
# ──────────────────────────────────────────────────────────────────────────────

class BaseEmbedder:
    """Shared interface for all embedding backends."""

    def __init__(self, dim: int):
        self.dim = dim

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise row-wise (needed for cosine similarity via inner product)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return vectors / norms


# ──────────────────────────────────────────────────────────────────────────────
#  Production: Sentence Transformers (multilingual)
# ──────────────────────────────────────────────────────────────────────────────

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Wraps the `sentence-transformers` library.
    Supports paraphrase-multilingual-mpnet-base-v2 (768-d, 50+ languages).

    Install:  pip install sentence-transformers
    """

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            cfg = CONFIG.embedding
            logger.info(f"Loading SentenceTransformer model: {cfg.model_name}")
            self._model = SentenceTransformer(cfg.model_name)
            self._model.max_seq_length = cfg.max_seq_length
            super().__init__(dim=cfg.embedding_dim)
            logger.info(f"Model loaded – dim={self.dim}")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        cfg = CONFIG.embedding
        all_vecs = []
        batch_size = cfg.batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            vecs  = self._model.encode(
                batch,
                batch_size        = batch_size,
                show_progress_bar = False,
                normalize_embeddings = cfg.normalize,
                convert_to_numpy  = True,
            )
            all_vecs.append(vecs)
        return np.vstack(all_vecs).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
#  Fallback: TF-IDF + LSA (works with only sklearn – zero external deps)
# ──────────────────────────────────────────────────────────────────────────────

class TFIDFEmbedder(BaseEmbedder):
    """
    Lightweight TF-IDF → TruncatedSVD (LSA) embedder.
    Used when sentence-transformers is unavailable (offline environments).

    • Multilingual: tokenises on Unicode word boundaries so it handles
      Arabic, Chinese, Cyrillic, etc. naturally.
    • dim defaults to min(300, n_features) to keep memory manageable.
    """

    def __init__(self, dim: int = 300):
        super().__init__(dim=dim)
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._svd:        Optional[TruncatedSVD]    = None
        self._fitted = False
        # Unicode-aware tokeniser (handles multilingual text)
        self._vectorizer = TfidfVectorizer(
            analyzer       = "word",
            token_pattern  = r"(?u)\b\w+\b",
            ngram_range    = (1, 2),
            max_features   = 50_000,
            sublinear_tf   = True,
            min_df         = 1,
        )
        self._svd = TruncatedSVD(n_components=dim, n_iter=5, random_state=42)

    def fit(self, texts: List[str]) -> "TFIDFEmbedder":
        """Fit on corpus – must be called before embed_texts."""
        logger.info(f"Fitting TF-IDF on {len(texts):,} documents …")
        tfidf_matrix = self._vectorizer.fit_transform(texts)
        # Clamp n_components to vocabulary size
        n_features = tfidf_matrix.shape[1]
        if self._svd.n_components >= n_features:
            self._svd = TruncatedSVD(
                n_components=max(1, n_features - 1),
                n_iter=5, random_state=42)
            self.dim = self._svd.n_components
        self._svd.fit(tfidf_matrix)
        self._fitted = True
        explained = self._svd.explained_variance_ratio_.sum()
        logger.info(f"TF-IDF fitted – vocab={n_features:,}, "
                    f"LSA dim={self.dim}, variance={explained:.2%}")
        return self

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            # Auto-fit on first call (single-shot mode)
            self.fit(texts)
        tfidf = self._vectorizer.transform(texts)
        vecs  = self._svd.transform(tfidf).astype(np.float32)
        return self._normalize(vecs)

    def embed_query(self, query: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TFIDFEmbedder must be fitted before embed_query.")
        tfidf = self._vectorizer.transform([query])
        vec   = self._svd.transform(tfidf).astype(np.float32)
        return self._normalize(vec)


# ──────────────────────────────────────────────────────────────────────────────
#  Unified Engine with caching
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Unified embedding interface that:
      1. Tries SentenceTransformer (production path).
      2. Falls back to TFIDFEmbedder (offline / CI path).
      3. Caches query embeddings to disk (optional).

    Usage
    -----
    engine = EmbeddingEngine()
    engine.fit_corpus(all_texts)        # required for TF-IDF backend
    vecs = engine.embed_texts(texts)    # → np.ndarray (N, dim)
    qvec = engine.embed_query(query)    # → np.ndarray (1, dim)
    """

    def __init__(self, prefer_transformer: bool = True):
        self._backend: BaseEmbedder
        self._cache: dict = {}
        self._cache_hits  = 0
        self._cache_misses= 0

        if prefer_transformer:
            try:
                self._backend = SentenceTransformerEmbedder()
                self._backend_name = "SentenceTransformer"
                logger.info("✓ Using SentenceTransformer backend (multilingual)")
            except Exception as e:
                logger.warning(f"SentenceTransformer unavailable ({e}). "
                               "Falling back to TF-IDF/LSA.")
                self._backend = TFIDFEmbedder(dim=300)
                self._backend_name = "TF-IDF/LSA"
        else:
            self._backend = TFIDFEmbedder(dim=300)
            self._backend_name = "TF-IDF/LSA"

    # ── corpus fitting (no-op for transformer) ───────────────────────────────
    def fit_corpus(self, texts: List[str]) -> "EmbeddingEngine":
        if isinstance(self._backend, TFIDFEmbedder):
            self._backend.fit(texts)
        return self

    # ── properties ───────────────────────────────────────────────────────────
    @property
    def dim(self) -> int:
        return self._backend.dim

    @property
    def backend_name(self) -> str:
        return self._backend_name

    # ── embed batch ──────────────────────────────────────────────────────────
    def embed_texts(self, texts: List[str],
                    batch_size: Optional[int] = None) -> np.ndarray:
        """Embed a list of texts; returns (N, dim) float32 array."""
        bs    = batch_size or CONFIG.embedding.batch_size
        parts = []
        t0    = time.time()
        for i in range(0, len(texts), bs):
            batch = texts[i: i + bs]
            parts.append(self._backend.embed_texts(batch))
        result = np.vstack(parts) if parts else np.empty((0, self.dim), dtype=np.float32)
        elapsed = time.time() - t0
        logger.debug(f"Embedded {len(texts):,} texts in {elapsed:.2f}s "
                     f"({len(texts)/max(elapsed,1e-9):.0f} texts/s)")
        return result

    # ── embed single query (with cache) ─────────────────────────────────────
    def embed_query(self, query: str) -> np.ndarray:
        key = hashlib.md5(query.encode()).hexdigest()
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        self._cache_misses += 1
        vec = self._backend.embed_query(query)
        if len(self._cache) < 10_000:
            self._cache[key] = vec
        return vec

    # ── diagnostics ──────────────────────────────────────────────────────────
    def cache_stats(self) -> dict:
        total = self._cache_hits + self._cache_misses
        return {
            "hits":      self._cache_hits,
            "misses":    self._cache_misses,
            "hit_rate":  self._cache_hits / max(total, 1),
            "cache_size": len(self._cache),
        }

    # ── cosine similarity helpers ────────────────────────────────────────────
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Scalar cosine similarity between two 1-D vectors."""
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a.flatten(), b.flatten()) / (na * nb))

    @staticmethod
    def batch_cosine_similarity(query_vec: np.ndarray,
                                doc_vecs: np.ndarray) -> np.ndarray:
        """Vectorised cosine similarity: query (1,d) vs docs (N,d) → (N,)."""
        q = query_vec.flatten()
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return np.zeros(len(doc_vecs))
        q = q / q_norm
        norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1e-10, norms)
        d_norm = doc_vecs / norms
        return (d_norm @ q).astype(np.float32)
