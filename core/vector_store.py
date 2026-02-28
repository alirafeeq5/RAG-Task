"""
Phase 1 – Vector Store (FAISS + Pure-NumPy Fallback)
=====================================================
Provides:
  • FAISSVectorStore      – production FAISS index (IVFFlat / Flat / HNSW)
  • NumpyVectorStore      – pure-NumPy fallback for offline environments
  • VectorStoreFactory    – picks the best available backend

Both stores expose the same interface:
    store.add(vectors, chunk_ids)
    results = store.search(query_vector, top_k)   → List[SearchResult]
    store.save() / store.load()
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import CONFIG, INDEX_DIR

logger = logging.getLogger(__name__)


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    chunk_id: str
    score:    float          # cosine similarity ∈ [-1, 1]
    rank:     int


# ── Base class ────────────────────────────────────────────────────────────────

class BaseVectorStore:
    def __init__(self, dim: int):
        self.dim    = dim
        self._count = 0

    def add(self, vectors: np.ndarray, chunk_ids: List[str]) -> None:
        raise NotImplementedError

    def search(self, query_vec: np.ndarray,
               top_k: int = 5) -> List[SearchResult]:
        raise NotImplementedError

    def save(self) -> None:
        raise NotImplementedError

    def load(self) -> None:
        raise NotImplementedError

    @property
    def size(self) -> int:
        return self._count


# ── Production: FAISS ────────────────────────────────────────────────────────

class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-backed vector store.
    Supports three index types:
      • "Flat"     – exact brute-force (small corpora)
      • "IVFFlat"  – inverted file index (large corpora, fast approximate)
      • "HNSW"     – hierarchical navigable small world (very fast approx)

    Install: pip install faiss-cpu
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise RuntimeError("FAISS not available. pip install faiss-cpu")

        cfg   = CONFIG.faiss
        d     = dim
        faiss = self._faiss

        if cfg.index_type == "IVFFlat":
            quantizer    = faiss.IndexFlatIP(d)
            self._index  = faiss.IndexIVFFlat(quantizer, d, cfg.nlist,
                                              faiss.METRIC_INNER_PRODUCT)
            self._index.nprobe = cfg.nprobe
            self._needs_train  = True
        elif cfg.index_type == "HNSW":
            self._index       = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
            self._needs_train  = False
        else:  # Flat – exact search
            self._index       = faiss.IndexFlatIP(d)
            self._needs_train  = False

        self._id_map: List[str] = []          # FAISS int64 → chunk_id

    def _train_if_needed(self, vectors: np.ndarray) -> None:
        if self._needs_train and not self._index.is_trained:
            nlist = CONFIG.faiss.nlist
            if len(vectors) < nlist * 39:     # FAISS requires ≥ 39×nlist train pts
                nlist = max(1, len(vectors) // 39)
                import faiss
                quantizer   = faiss.IndexFlatIP(self.dim)
                self._index = faiss.IndexIVFFlat(quantizer, self.dim, nlist,
                                                 faiss.METRIC_INNER_PRODUCT)
                self._index.nprobe = min(CONFIG.faiss.nprobe, nlist)
            logger.info(f"Training IVFFlat index with {len(vectors)} vectors …")
            self._index.train(vectors)

    def add(self, vectors: np.ndarray, chunk_ids: List[str]) -> None:
        vecs = vectors.astype(np.float32)
        self._train_if_needed(vecs)
        self._index.add(vecs)
        self._id_map.extend(chunk_ids)
        self._count += len(chunk_ids)
        logger.debug(f"FAISS add: +{len(chunk_ids)} → total {self._count}")

    def search(self, query_vec: np.ndarray,
               top_k: int = 5) -> List[SearchResult]:
        q = query_vec.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._count)
        if k == 0:
            return []
        scores, indices = self._index.search(q, k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append(SearchResult(
                chunk_id = self._id_map[idx],
                score    = float(score),
                rank     = rank,
            ))
        return results

    def save(self) -> None:
        import faiss
        path = CONFIG.faiss.index_path
        faiss.write_index(self._index, str(path))
        meta = {"dim": self.dim, "count": self._count, "id_map": self._id_map}
        with open(str(path) + ".meta.json", "w") as f:
            json.dump(meta, f)
        logger.info(f"FAISS index saved to {path}")

    def load(self) -> None:
        import faiss
        path = CONFIG.faiss.index_path
        self._index = faiss.read_index(str(path))
        with open(str(path) + ".meta.json") as f:
            meta = json.load(f)
        self._id_map = meta["id_map"]
        self._count  = meta["count"]
        logger.info(f"FAISS index loaded – {self._count:,} vectors")


# ── Fallback: Pure NumPy (offline environments) ──────────────────────────────

class NumpyVectorStore(BaseVectorStore):
    """
    Brute-force cosine similarity search using NumPy.
    No external dependencies; suitable for small-to-medium corpora.
    At 300-dim float32:  100k vectors ≈ 120 MB RAM, search ≈ 50 ms.
    """

    def __init__(self, dim: int):
        super().__init__(dim)
        self._vectors: Optional[np.ndarray] = None  # (N, dim) float32
        self._id_map:  List[str]            = []
        self._index_path = INDEX_DIR / "numpy_index.npz"
        self._meta_path  = INDEX_DIR / "numpy_meta.json"

    def add(self, vectors: np.ndarray, chunk_ids: List[str]) -> None:
        vecs = vectors.astype(np.float32)
        if self._vectors is None:
            self._vectors = vecs
        else:
            self._vectors = np.vstack([self._vectors, vecs])
        self._id_map.extend(chunk_ids)
        self._count += len(chunk_ids)
        logger.debug(f"NumPy store add: +{len(chunk_ids)} → total {self._count}")

    def search(self, query_vec: np.ndarray,
               top_k: int = 5) -> List[SearchResult]:
        if self._vectors is None or self._count == 0:
            return []
        q = query_vec.flatten().astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-10:
            return []
        q = q / q_norm

        # Vectorised cosine similarity
        norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1e-10, norms)
        normed = self._vectors / norms
        scores = (normed @ q).astype(np.float32)           # (N,)

        k = min(top_k, self._count)
        top_indices = np.argpartition(scores, -k)[-k:]     # fast top-k
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [
            SearchResult(
                chunk_id = self._id_map[i],
                score    = float(scores[i]),
                rank     = rank,
            )
            for rank, i in enumerate(top_indices)
            if 0 <= i < len(self._id_map)
        ]

    def save(self) -> None:
        if self._vectors is not None:
            np.savez_compressed(str(self._index_path), vectors=self._vectors)
        with open(str(self._meta_path), "w") as f:
            json.dump({"dim": self.dim, "count": self._count,
                       "id_map": self._id_map}, f)
        logger.info(f"NumPy index saved ({self._count:,} vectors)")

    def load(self) -> None:
        if self._index_path.exists():
            data = np.load(str(self._index_path))
            self._vectors = data["vectors"]
        with open(str(self._meta_path)) as f:
            meta = json.load(f)
        self._id_map = meta["id_map"]
        self._count  = meta["count"]
        logger.info(f"NumPy index loaded – {self._count:,} vectors")


# ── Factory ───────────────────────────────────────────────────────────────────

class VectorStoreFactory:
    """Returns the best available vector store for the given dimensionality."""

    @staticmethod
    def create(dim: int, prefer_faiss: bool = True) -> BaseVectorStore:
        if prefer_faiss:
            try:
                store = FAISSVectorStore(dim)
                logger.info(f"✓ Using FAISS vector store (dim={dim})")
                return store
            except Exception as e:
                logger.warning(f"FAISS unavailable ({e}). Using NumPy fallback.")
        store = NumpyVectorStore(dim)
        logger.info(f"✓ Using NumPy vector store (dim={dim})")
        return store


# ── Document Index (embedding + store bundled) ────────────────────────────────

class DocumentIndex:
    """
    High-level index that ties together:
      EmbeddingEngine  ──►  VectorStore
                       and  metadata lookup dict (chunk_id → DocumentChunk)

    Usage
    -----
    idx = DocumentIndex(engine, store)
    idx.index_chunks(chunks)          # add chunks
    results = idx.search("query", k=5)
    """

    def __init__(self, embedding_engine, vector_store: BaseVectorStore):
        self._engine  = embedding_engine
        self._store   = vector_store
        self._meta:   Dict[str, dict] = {}    # chunk_id → serialisable metadata
        self._text:   Dict[str, str]  = {}    # chunk_id → text

    def index_chunks(self, chunks, batch_size: int = 256) -> None:
        """Embed and index a list of DocumentChunk objects."""
        from core.data_processor import DocumentChunk
        logger.info(f"Indexing {len(chunks):,} chunks …")
        t0 = time.time()

        all_texts    = [c.text for c in chunks]
        all_ids      = [c.chunk_id for c in chunks]

        # fit corpus for TF-IDF backend before embedding
        self._engine.fit_corpus(all_texts)

        for i in range(0, len(chunks), batch_size):
            batch  = chunks[i: i + batch_size]
            texts  = [c.text for c in batch]
            ids    = [c.chunk_id for c in batch]
            vecs   = self._engine.embed_texts(texts)
            self._store.add(vecs, ids)
            for c in batch:
                self._meta[c.chunk_id] = c.metadata
                self._text[c.chunk_id] = c.text

        elapsed = time.time() - t0
        logger.info(f"Indexing complete – {len(chunks):,} chunks in {elapsed:.1f}s")

    def search(self, query: str,
               top_k: int = None,
               score_threshold: float = None) -> List[dict]:
        """
        Retrieve top-k chunks for a query.
        Returns list of dicts: {chunk_id, text, score, rank, metadata}
        """
        k    = top_k or CONFIG.retrieval.top_k
        thr  = score_threshold if score_threshold is not None \
               else CONFIG.retrieval.score_threshold

        q_vec   = self._engine.embed_query(query)
        raw_res = self._store.search(q_vec, top_k=k)

        results = []
        for r in raw_res:
            if r.score < thr:
                continue
            results.append({
                "chunk_id": r.chunk_id,
                "text":     self._text.get(r.chunk_id, ""),
                "score":    r.score,
                "rank":     r.rank,
                "metadata": self._meta.get(r.chunk_id, {}),
            })
        return results

    def save(self) -> None:
        self._store.save()
        meta_path = INDEX_DIR / "doc_index_meta.json"
        with open(meta_path, "w") as f:
            json.dump({"meta": self._meta, "text": self._text}, f)
        logger.info("DocumentIndex saved.")

    def load(self) -> None:
        self._store.load()
        meta_path = INDEX_DIR / "doc_index_meta.json"
        with open(meta_path) as f:
            data = json.load(f)
        self._meta = data["meta"]
        self._text = data["text"]
        logger.info("DocumentIndex loaded.")

    @property
    def size(self) -> int:
        return self._store.size
