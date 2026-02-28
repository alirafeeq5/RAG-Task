"""
Database Layer – SQLAlchemy Models & Repository
================================================
Tables:
  • documents   – enriched records from the dataset
  • chunks      – document chunks with metadata
  • queries     – logged user queries
  • responses   – generated answers with quality metrics
  • conversations – conversation sessions
"""

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

# SQLAlchemy (stdlib alternative: sqlite3 – used as fallback below)
try:
    from sqlalchemy import (
        create_engine, Column, Integer, String, Float, Boolean,
        Text, DateTime, ForeignKey, JSON, Index, func
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, Session, relationship
    from sqlalchemy.exc import SQLAlchemyError
    _SA_AVAILABLE = True
except ImportError:
    _SA_AVAILABLE = False

import sqlite3

from config import CONFIG, DB_PATH

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  SQLAlchemy Models
# ──────────────────────────────────────────────────────────────────────────────

if _SA_AVAILABLE:
    Base = declarative_base()

    class DocumentModel(Base):
        __tablename__ = "documents"
        id            = Column(Integer, primary_key=True, autoincrement=True)
        record_id     = Column(String(32), unique=True, nullable=False, index=True)
        row_id        = Column(Integer)
        question      = Column(Text, nullable=False)
        long_answer   = Column(Text)
        short_answer  = Column(Text)
        has_short     = Column(Boolean, default=False)
        has_long      = Column(Boolean, default=False)
        question_type = Column(String(20), index=True)
        domain        = Column(String(50), index=True)
        difficulty    = Column(String(10), index=True)
        answer_length = Column(Integer, default=0)
        language      = Column(String(20), default="en")
        created_at    = Column(Float, default=time.time)

        chunks        = relationship("ChunkModel", back_populates="document",
                                     cascade="all, delete-orphan")

    class ChunkModel(Base):
        __tablename__ = "chunks"
        id          = Column(Integer, primary_key=True, autoincrement=True)
        chunk_id    = Column(String(32), unique=True, nullable=False, index=True)
        record_id   = Column(String(32), ForeignKey("documents.record_id"),
                             nullable=False)
        row_id      = Column(Integer)
        question    = Column(Text)
        text        = Column(Text, nullable=False)
        chunk_index = Column(Integer, default=0)
        chunk_type  = Column(String(30), index=True)
        token_count = Column(Integer, default=0)
        metadata_   = Column("metadata", Text)  # JSON string
        created_at  = Column(Float, default=time.time)

        document    = relationship("DocumentModel", back_populates="chunks")

    class QueryLogModel(Base):
        __tablename__  = "query_logs"
        id             = Column(Integer, primary_key=True, autoincrement=True)
        session_id     = Column(String(64), index=True)
        raw_query      = Column(Text, nullable=False)
        processed_query= Column(Text)
        expanded_queries = Column(Text)   # JSON list
        num_results    = Column(Integer, default=0)
        elapsed_ms     = Column(Float, default=0)
        created_at     = Column(Float, default=time.time)

    class ResponseLogModel(Base):
        __tablename__ = "response_logs"
        id            = Column(Integer, primary_key=True, autoincrement=True)
        query_log_id  = Column(Integer, ForeignKey("query_logs.id"))
        answer        = Column(Text)
        is_valid      = Column(Boolean, default=False)
        confidence    = Column(Float, default=0.0)
        used_fallback = Column(Boolean, default=False)
        quality_issues= Column(Text)   # JSON list
        latency_ms    = Column(Float, default=0)
        source_chunks = Column(Text)   # JSON list
        created_at    = Column(Float, default=time.time)


# ──────────────────────────────────────────────────────────────────────────────
#  Database Manager
# ──────────────────────────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Thin repository layer over SQLAlchemy.
    Falls back to raw sqlite3 if SQLAlchemy is not installed.
    """

    def __init__(self):
        if _SA_AVAILABLE:
            self._engine = create_engine(
                CONFIG.database.url,
                echo       = CONFIG.database.echo_sql,
                connect_args = {"check_same_thread": False},
            )
            Base.metadata.create_all(self._engine)
            self._SessionFactory = sessionmaker(bind=self._engine)
            logger.info(f"SQLAlchemy DB ready at {CONFIG.database.url}")
        else:
            self._conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
            self._create_tables_raw()
            logger.info(f"sqlite3 DB ready at {DB_PATH}")

    # ── session context manager ───────────────────────────────────────────────
    def _session(self):
        if _SA_AVAILABLE:
            return self._SessionFactory()
        return None

    # ── raw sqlite3 fallback ──────────────────────────────────────────────────
    def _create_tables_raw(self) -> None:
        c = self._conn.cursor()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id TEXT UNIQUE NOT NULL,
            row_id INTEGER,
            question TEXT NOT NULL,
            long_answer TEXT,
            short_answer TEXT,
            has_short INTEGER DEFAULT 0,
            has_long INTEGER DEFAULT 0,
            question_type TEXT,
            domain TEXT,
            difficulty TEXT,
            answer_length INTEGER DEFAULT 0,
            language TEXT DEFAULT 'en',
            created_at REAL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id TEXT UNIQUE NOT NULL,
            record_id TEXT NOT NULL,
            row_id INTEGER,
            question TEXT,
            text TEXT NOT NULL,
            chunk_index INTEGER DEFAULT 0,
            chunk_type TEXT,
            token_count INTEGER DEFAULT 0,
            metadata TEXT,
            created_at REAL
        );
        CREATE TABLE IF NOT EXISTS query_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            raw_query TEXT NOT NULL,
            processed_query TEXT,
            expanded_queries TEXT,
            num_results INTEGER DEFAULT 0,
            elapsed_ms REAL DEFAULT 0,
            created_at REAL
        );
        CREATE TABLE IF NOT EXISTS response_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_log_id INTEGER,
            answer TEXT,
            is_valid INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0,
            used_fallback INTEGER DEFAULT 0,
            quality_issues TEXT,
            latency_ms REAL DEFAULT 0,
            source_chunks TEXT,
            created_at REAL
        );
        """)
        self._conn.commit()

    # ── Document CRUD ─────────────────────────────────────────────────────────
    def save_documents(self, enriched_records: List) -> int:
        """Bulk-insert enriched records; skip duplicates."""
        rows = [asdict(r) for r in enriched_records]
        saved = 0
        if _SA_AVAILABLE:
            session = self._session()
            try:
                for row in rows:
                    exists = (session.query(DocumentModel)
                              .filter_by(record_id=row["record_id"]).first())
                    if not exists:
                        doc = DocumentModel(**{k: v for k, v in row.items()
                                               if k not in ("long_answer","short_answer")})
                        doc.long_answer  = row["long_answer"]
                        doc.short_answer = row["short_answer"]
                        session.add(doc)
                        saved += 1
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"DB save_documents error: {e}")
            finally:
                session.close()
        else:
            c = self._conn.cursor()
            for row in rows:
                try:
                    c.execute("""
                        INSERT OR IGNORE INTO documents
                        (record_id,row_id,question,long_answer,short_answer,
                         has_short,has_long,question_type,domain,difficulty,
                         answer_length,language,created_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                        row["record_id"], row["row_id"], row["question"],
                        row["long_answer"], row["short_answer"],
                        int(row["has_short"]), int(row["has_long"]),
                        row["question_type"], row["domain"], row["difficulty"],
                        row["answer_length"], row["language"], time.time(),
                    ))
                    saved += c.rowcount
                except sqlite3.IntegrityError:
                    pass
            self._conn.commit()
        return saved

    # ── Chunk CRUD ────────────────────────────────────────────────────────────
    def save_chunks(self, chunks: List) -> int:
        """Bulk-insert DocumentChunk objects; skip duplicates."""
        saved = 0
        if _SA_AVAILABLE:
            session = self._session()
            try:
                for c in chunks:
                    exists = (session.query(ChunkModel)
                              .filter_by(chunk_id=c.chunk_id).first())
                    if not exists:
                        session.add(ChunkModel(
                            chunk_id    = c.chunk_id,
                            record_id   = c.record_id,
                            row_id      = c.row_id,
                            question    = c.question,
                            text        = c.text,
                            chunk_index = c.chunk_index,
                            chunk_type  = c.chunk_type,
                            token_count = c.token_count,
                            metadata_   = json.dumps(c.metadata),
                            created_at  = time.time(),
                        ))
                        saved += 1
                session.commit()
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"DB save_chunks error: {e}")
            finally:
                session.close()
        else:
            cur = self._conn.cursor()
            for c in chunks:
                try:
                    cur.execute("""
                        INSERT OR IGNORE INTO chunks
                        (chunk_id,record_id,row_id,question,text,
                         chunk_index,chunk_type,token_count,metadata,created_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?)""", (
                        c.chunk_id, c.record_id, c.row_id, c.question,
                        c.text, c.chunk_index, c.chunk_type, c.token_count,
                        json.dumps(c.metadata), time.time(),
                    ))
                    saved += cur.rowcount
                except sqlite3.IntegrityError:
                    pass
            self._conn.commit()
        return saved

    # ── Query / Response logging ──────────────────────────────────────────────
    def log_query(self, session_id: str, raw_query: str,
                  processed_query: str, expanded: List[str],
                  num_results: int, elapsed_ms: float) -> int:
        """Insert a query log entry; return new row id."""
        if _SA_AVAILABLE:
            session = self._session()
            try:
                row = QueryLogModel(
                    session_id       = session_id,
                    raw_query        = raw_query,
                    processed_query  = processed_query,
                    expanded_queries = json.dumps(expanded),
                    num_results      = num_results,
                    elapsed_ms       = elapsed_ms,
                    created_at       = time.time(),
                )
                session.add(row)
                session.commit()
                return row.id
            finally:
                session.close()
        else:
            cur = self._conn.cursor()
            cur.execute("""
                INSERT INTO query_logs
                (session_id,raw_query,processed_query,expanded_queries,
                 num_results,elapsed_ms,created_at)
                VALUES (?,?,?,?,?,?,?)""", (
                session_id, raw_query, processed_query,
                json.dumps(expanded), num_results, elapsed_ms, time.time()
            ))
            self._conn.commit()
            return cur.lastrowid

    def log_response(self, query_log_id: int, answer: str,
                     is_valid: bool, confidence: float,
                     used_fallback: bool, issues: List[str],
                     latency_ms: float, source_chunks: List[str]) -> None:
        if _SA_AVAILABLE:
            session = self._session()
            try:
                row = ResponseLogModel(
                    query_log_id  = query_log_id,
                    answer        = answer,
                    is_valid      = is_valid,
                    confidence    = confidence,
                    used_fallback = used_fallback,
                    quality_issues= json.dumps(issues),
                    latency_ms    = latency_ms,
                    source_chunks = json.dumps(source_chunks),
                    created_at    = time.time(),
                )
                session.add(row)
                session.commit()
            finally:
                session.close()
        else:
            cur = self._conn.cursor()
            cur.execute("""
                INSERT INTO response_logs
                (query_log_id,answer,is_valid,confidence,used_fallback,
                 quality_issues,latency_ms,source_chunks,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)""", (
                query_log_id, answer, int(is_valid), confidence,
                int(used_fallback), json.dumps(issues),
                latency_ms, json.dumps(source_chunks), time.time()
            ))
            self._conn.commit()

    # ── Stats ────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        if _SA_AVAILABLE:
            session = self._session()
            try:
                return {
                    "documents":  session.query(DocumentModel).count(),
                    "chunks":     session.query(ChunkModel).count(),
                    "queries":    session.query(QueryLogModel).count(),
                    "responses":  session.query(ResponseLogModel).count(),
                    "avg_confidence": session.query(
                        func.avg(ResponseLogModel.confidence)).scalar() or 0,
                }
            finally:
                session.close()
        else:
            cur = self._conn.cursor()
            def count(t): return cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            avg_c = cur.execute(
                "SELECT AVG(confidence) FROM response_logs").fetchone()[0] or 0
            return {
                "documents":       count("documents"),
                "chunks":          count("chunks"),
                "queries":         count("query_logs"),
                "responses":       count("response_logs"),
                "avg_confidence":  round(avg_c, 3),
            }
