"""
FastAPI REST API
================
Endpoints:
  POST /query              – single-turn RAG query
  POST /conversation/query – multi-turn conversational RAG
  DELETE /conversation/{id}– reset conversation context
  GET  /health             – system health check
  GET  /stats              – system metrics
  POST /index/rebuild      – trigger re-indexing (admin)
  GET  /docs               – auto-generated Swagger UI (built-in FastAPI)

Note: FastAPI is imported conditionally; if unavailable the module
still exports RAGPipeline which can be used directly.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Pydantic Models ───────────────────────────────────────────────────────────
try:
    from pydantic import BaseModel, Field

    class QueryRequest(BaseModel):
        query:           str   = Field(..., min_length=1, max_length=2000)
        top_k:           int   = Field(default=5, ge=1, le=20)
        score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
        use_expansion:   bool  = True
        use_rerank:      bool  = True
        session_id:      Optional[str] = None

    class ConvQueryRequest(QueryRequest):
        session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    class QueryResponse(BaseModel):
        answer:          str
        confidence:      float
        is_valid:        bool
        used_fallback:   bool
        retrieved_chunks: List[Dict[str, Any]]
        expanded_queries: List[str]
        latency_ms:      float
        session_id:      Optional[str] = None

    _PYDANTIC_OK = True
except ImportError:
    _PYDANTIC_OK = False


# ── RAG Pipeline (framework-agnostic core) ───────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import CONFIG
from core.embeddings       import EmbeddingEngine
from core.vector_store     import VectorStoreFactory, DocumentIndex
from core.query_processor  import QueryProcessor
from core.response_generator import ResponseGenerator
from database.models       import DatabaseManager
from utils.performance     import PerformanceMonitor, BatchProcessor


class RAGPipeline:
    """
    Top-level orchestrator that wires all components together.

    Lifecycle
    ---------
    pipeline = RAGPipeline()
    pipeline.build(chunks)           # index documents
    response = pipeline.query("…")   # answer a question
    """

    def __init__(self):
        logger.info("Initialising RAG Pipeline …")
        self._engine    = EmbeddingEngine(prefer_transformer=True)
        self._store     = VectorStoreFactory.create(self._engine.dim)
        self._index     = DocumentIndex(self._engine, self._store)
        self._processor = QueryProcessor(self._index, self._engine)
        self._generator = ResponseGenerator()
        self._db        = DatabaseManager()
        self._monitor   = PerformanceMonitor()
        self._sessions: Dict[str, QueryProcessor] = {}  # per-session processors
        self._ready     = False

    # ── indexing ──────────────────────────────────────────────────────────────
    def build(self, chunks: List, batch_size: int = 256) -> "RAGPipeline":
        """Embed and index all DocumentChunk objects."""
        bp = BatchProcessor(
            self._engine, self._store,
            batch_size = batch_size,
            monitor    = self._monitor,
        )
        # build metadata/text lookup inside DocumentIndex directly
        self._index.index_chunks(chunks, batch_size=batch_size)
        self._ready = True
        logger.info(f"RAG Pipeline ready – {self._index.size:,} chunks indexed.")
        return self

    def save_index(self) -> None:
        self._index.save()

    def load_index(self) -> None:
        self._index.load()
        self._ready = True

    # ── querying ──────────────────────────────────────────────────────────────
    def query(self,
              question:       str,
              top_k:          int   = None,
              score_threshold: float = None,
              use_expansion:  bool  = True,
              use_rerank:     bool  = True,
              use_conversation: bool = False,
              session_id:     str   = None) -> Dict:
        """
        Full RAG query pipeline.

        Returns
        -------
        {
          answer, confidence, is_valid, used_fallback,
          retrieved, expanded_queries, latency_ms, session_id
        }
        """
        if not self._ready:
            return {"answer": "System is not ready – please index documents first.",
                    "confidence": 0.0, "is_valid": False, "used_fallback": True,
                    "retrieved": [], "expanded_queries": [], "latency_ms": 0}

        t0 = time.time()

        # Per-session processor for conversation context
        if use_conversation and session_id:
            if session_id not in self._sessions:
                self._sessions[session_id] = QueryProcessor(
                    self._index, self._engine)
            processor = self._sessions[session_id]
        else:
            processor = self._processor

        # Retrieve
        with self._monitor.track("retrieve"):
            ret = processor.process(
                question,
                top_k            = top_k,
                score_threshold  = score_threshold,
                use_expansion    = use_expansion,
                use_rerank       = use_rerank,
                use_conversation = use_conversation and bool(session_id),
            )

        # Detect language of query
        from core.data_processor import detect_language
        lang = detect_language(question)

        # Generate
        with self._monitor.track("generate"):
            history = (processor.conversation_history
                       if use_conversation and session_id else None)
            gen_result = self._generator.generate(
                query     = ret["processed_query"],
                contexts  = ret["results"],
                history   = history,
                language  = lang,
            )

        # Update conversation history
        if use_conversation and session_id and session_id in self._sessions:
            self._sessions[session_id].add_to_history("user", question)
            self._sessions[session_id].add_to_history(
                "assistant", gen_result.answer)

        # Log to DB (non-blocking, best-effort)
        try:
            qid = self._db.log_query(
                session_id      = session_id or "anon",
                raw_query       = question,
                processed_query = ret["processed_query"],
                expanded        = ret["expanded_queries"],
                num_results     = len(ret["results"]),
                elapsed_ms      = ret["elapsed_ms"],
            )
            self._db.log_response(
                query_log_id  = qid,
                answer        = gen_result.answer,
                is_valid      = gen_result.is_valid,
                confidence    = gen_result.confidence,
                used_fallback = gen_result.used_fallback,
                issues        = gen_result.quality_issues,
                latency_ms    = gen_result.latency_ms,
                source_chunks = gen_result.source_chunks,
            )
        except Exception:
            pass

        total_ms = round((time.time() - t0) * 1000, 1)
        return {
            "answer":           gen_result.answer,
            "confidence":       gen_result.confidence,
            "is_valid":         gen_result.is_valid,
            "used_fallback":    gen_result.used_fallback,
            "retrieved":        ret["results"],
            "expanded_queries": ret["expanded_queries"],
            "latency_ms":       total_ms,
            "session_id":       session_id,
        }

    def reset_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

    # ── system info ───────────────────────────────────────────────────────────
    def health(self) -> Dict:
        return {
            "status":        "ready" if self._ready else "not_indexed",
            "backend":       self._engine.backend_name,
            "index_size":    self._index.size,
            "db_stats":      self._db.get_stats(),
            "perf_metrics":  self._monitor.report(),
            "cache_stats":   self._engine.cache_stats(),
            "llm_stats":     self._generator.stats,
        }


# ── FastAPI App ───────────────────────────────────────────────────────────────

def create_app(pipeline: RAGPipeline):
    """
    Create a FastAPI application wired to the given RAGPipeline.
    Import and call this only in environments where fastapi is installed.
    """
    try:
        from fastapi import FastAPI, HTTPException, BackgroundTasks
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise RuntimeError("FastAPI not available. pip install fastapi uvicorn")

    app = FastAPI(
        title       = "Multilingual RAG API",
        description = "Production RAG system with FAISS + Sentence Transformers + Groq",
        version     = "1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins  = ["*"],
        allow_methods  = ["*"],
        allow_headers  = ["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return pipeline.health()

    @app.get("/stats")
    async def stats():
        return pipeline.health()

    @app.post("/query", response_model=QueryResponse if _PYDANTIC_OK else None)
    async def query(req: QueryRequest):
        result = pipeline.query(
            question        = req.query,
            top_k           = req.top_k,
            score_threshold = req.score_threshold,
            use_expansion   = req.use_expansion,
            use_rerank      = req.use_rerank,
            use_conversation= False,
        )
        if _PYDANTIC_OK:
            return QueryResponse(
                answer           = result["answer"],
                confidence       = result["confidence"],
                is_valid         = result["is_valid"],
                used_fallback    = result["used_fallback"],
                retrieved_chunks = result["retrieved"],
                expanded_queries = result["expanded_queries"],
                latency_ms       = result["latency_ms"],
            )
        return result

    @app.post("/conversation/query")
    async def conv_query(req: ConvQueryRequest):
        session_id = req.session_id or str(uuid.uuid4())
        result     = pipeline.query(
            question        = req.query,
            top_k           = req.top_k,
            score_threshold = req.score_threshold,
            use_expansion   = req.use_expansion,
            use_rerank      = req.use_rerank,
            use_conversation= True,
            session_id      = session_id,
        )
        result["session_id"] = session_id
        return result

    @app.delete("/conversation/{session_id}")
    async def reset_conversation(session_id: str):
        pipeline.reset_session(session_id)
        return {"message": f"Session {session_id} reset."}

    @app.post("/index/rebuild")
    async def rebuild_index(background_tasks: BackgroundTasks):
        return {"message": "Index rebuild not exposed in demo mode."}

    return app


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick CLI test
    import sys
    sys.path.insert(0, "..")
    try:
        import uvicorn
        from main import pipeline
        app = create_app(pipeline)
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("Run via main.py instead.")
