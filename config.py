"""
Configuration module for the Multilingual RAG System.
All system-wide settings, paths, and hyperparameters live here.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# ── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
INDEX_DIR  = BASE_DIR / "indexes"
CACHE_DIR  = BASE_DIR / "cache"
LOG_DIR    = BASE_DIR / "logs"
DB_PATH    = BASE_DIR / "rag_system.db"

for _d in (DATA_DIR, INDEX_DIR, CACHE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Embedding Settings ─────────────────────────────────────────────────────────
@dataclass
class EmbeddingConfig:
    # Primary multilingual model – covers 50+ languages
    model_name:     str   = "paraphrase-multilingual-mpnet-base-v2"
    # Fallback English-only model
    fallback_model: str   = "all-MiniLM-L6-v2"
    embedding_dim:  int   = 768          # output dimension of the primary model
    batch_size:     int   = 64           # documents per encoding batch
    max_seq_length: int   = 512          # token limit per document chunk
    normalize:      bool  = True         # L2-normalise embeddings


# ── FAISS Vector Store Settings ───────────────────────────────────────────────
@dataclass
class FAISSConfig:
    index_type:    str   = "IVFFlat"     # IVFFlat | Flat | HNSW
    nlist:         int   = 100           # IVF: number of Voronoi cells
    nprobe:        int   = 10            # IVF: cells to visit at query time
    metric:        str   = "cosine"      # cosine | l2 | ip
    index_path:    Path  = INDEX_DIR / "faiss_index.bin"
    metadata_path: Path  = INDEX_DIR / "metadata.json"


# ── Chunking Settings ─────────────────────────────────────────────────────────
@dataclass
class ChunkingConfig:
    strategy:         str  = "hybrid"   # fixed | sentence | hybrid
    max_chunk_tokens: int  = 256        # max tokens in one chunk
    overlap_tokens:   int  = 32         # overlap between consecutive chunks
    min_chunk_tokens: int  = 20         # discard chunks shorter than this


# ── Retrieval Settings ────────────────────────────────────────────────────────
@dataclass
class RetrievalConfig:
    top_k:               int   = 5
    score_threshold:     float = 0.35   # minimum similarity score
    rerank:              bool  = True   # enable cross-encoder re-ranking
    mmr_lambda:          float = 0.7    # Maximal Marginal Relevance diversity
    context_window:      int   = 3      # number of past turns to keep


# ── LLM / Generation Settings ────────────────────────────────────────────────
@dataclass
class LLMConfig:
    provider:        str   = "groq"
    api_key:         str   = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
    base_url:        str   = "https://api.groq.com/openai/v1"
    model:           str   = "llama3-8b-8192"
    max_tokens:      int   = 512
    temperature:     float = 0.2
    timeout_seconds: int   = 30
    # Quality gate
    min_answer_len:  int   = 10
    fallback_msg:    str   = (
        "I couldn't find a confident answer in the knowledge base. "
        "Please try rephrasing your question or providing more context."
    )


# ── Caching Settings ──────────────────────────────────────────────────────────
@dataclass
class CacheConfig:
    enabled:        bool  = True
    max_size:       int   = 1_000       # max cached query→result entries
    ttl_seconds:    int   = 3_600       # time-to-live per entry (1 hour)
    cache_dir:      Path  = CACHE_DIR


# ── Database Settings ─────────────────────────────────────────────────────────
@dataclass
class DatabaseConfig:
    url:        str  = f"sqlite:///{DB_PATH}"
    echo_sql:   bool = False
    pool_size:  int  = 5


# ── Logging Settings ──────────────────────────────────────────────────────────
@dataclass
class LoggingConfig:
    level:    str  = "INFO"
    log_dir:  Path = LOG_DIR
    filename: str  = "rag_system.log"


# ── Master Config ─────────────────────────────────────────────────────────────
@dataclass
class RAGConfig:
    embedding:  EmbeddingConfig  = field(default_factory=EmbeddingConfig)
    faiss:      FAISSConfig      = field(default_factory=FAISSConfig)
    chunking:   ChunkingConfig   = field(default_factory=ChunkingConfig)
    retrieval:  RetrievalConfig  = field(default_factory=RetrievalConfig)
    llm:        LLMConfig        = field(default_factory=LLMConfig)
    cache:      CacheConfig      = field(default_factory=CacheConfig)
    database:   DatabaseConfig   = field(default_factory=DatabaseConfig)
    logging:    LoggingConfig    = field(default_factory=LoggingConfig)
    debug:      bool             = False


# Singleton used everywhere
CONFIG = RAGConfig()
