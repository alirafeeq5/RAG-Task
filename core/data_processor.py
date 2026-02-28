"""
Phase 1 – Dataset Acquisition & Processing
==========================================
Handles:
  • Loading the Natural Questions CSV (sample or full dataset)
  • HTML stripping and text normalisation
  • Metadata enrichment: question_type, domain, difficulty
  • Hybrid text-chunking (fixed-size with sentence-boundary awareness)
  • Short-answer vs long-answer routing
"""

import re
import json
import logging
import hashlib
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple

import pandas as pd
import numpy as np

from config import CONFIG, DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class RawRecord:
    """One row from the Natural Questions CSV."""
    row_id:        int
    question:      str
    long_answer:   str
    short_answer:  str


@dataclass
class EnrichedRecord:
    """Raw record augmented with domain / type / difficulty metadata."""
    record_id:     str          # sha256 of question
    row_id:        int
    question:      str
    long_answer:   str
    short_answer:  str
    has_short:     bool
    has_long:      bool
    question_type: str          # what / who / when / where / how / why / other
    domain:        str          # science / history / entertainment / sports / …
    difficulty:    str          # easy / medium / hard
    answer_length: int          # word-count of long answer
    language:      str          # detected language code (e.g. "en")


@dataclass
class DocumentChunk:
    """A retrievable chunk of text derived from an EnrichedRecord."""
    chunk_id:      str          # unique deterministic id
    record_id:     str
    row_id:        int
    question:      str
    text:          str          # actual chunk content
    chunk_index:   int          # position within the record
    chunk_type:    str          # "short_answer" | "long_answer_chunk"
    metadata:      Dict         # all EnrichedRecord fields except raw text
    token_count:   int


# ── HTML & Text Cleaning ──────────────────────────────────────────────────────

_HTML_TAG   = re.compile(r"<[^>]+>")
_WS_MULTI   = re.compile(r"\s+")
_PUNCT_NORM = re.compile(r"\s+([,.;:!?])")

def clean_text(text: str) -> str:
    """Strip HTML, collapse whitespace, normalise punctuation."""
    if not isinstance(text, str):
        return ""
    text = _HTML_TAG.sub(" ", text)               # remove HTML tags
    text = text.replace("``", '"').replace("''", '"')  # typographic quotes
    text = _PUNCT_NORM.sub(r"\1", text)           # remove space before punct
    text = _WS_MULTI.sub(" ", text)               # collapse whitespace
    return text.strip()


# ── Language Detection (lightweight, no external library) ────────────────────

# Simple heuristic: look for non-ASCII characters to detect non-English text.
_LATIN_EXTENDED = re.compile(r"[À-ÿ]")
_ARABIC         = re.compile(r"[\u0600-\u06FF]")
_CJK            = re.compile(r"[\u4E00-\u9FFF]")
_CYRILLIC       = re.compile(r"[\u0400-\u04FF]")

def detect_language(text: str) -> str:
    """Lightweight language detection without external dependencies."""
    if _ARABIC.search(text):   return "ar"
    if _CJK.search(text):      return "zh"
    if _CYRILLIC.search(text): return "ru"
    if _LATIN_EXTENDED.search(text): return "es_fr_de"  # broad Romance/Germanic
    return "en"


# ── Question-Type Classification ─────────────────────────────────────────────

_Q_TYPE_PATTERNS = {
    "what":  re.compile(r"\bwhat\b", re.I),
    "who":   re.compile(r"\bwho\b",  re.I),
    "when":  re.compile(r"\bwhen\b", re.I),
    "where": re.compile(r"\bwhere\b",re.I),
    "how":   re.compile(r"\bhow\b",  re.I),
    "why":   re.compile(r"\bwhy\b",  re.I),
    "which": re.compile(r"\bwhich\b",re.I),
    "is_are":re.compile(r"^(is|are|was|were|does|do|did|can|could|will|would)\b", re.I),
}

def classify_question_type(question: str) -> str:
    for q_type, pattern in _Q_TYPE_PATTERNS.items():
        if pattern.search(question):
            return q_type
    return "other"


# ── Domain Classification ─────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "science":      ["biology","chemistry","physics","science","scientific",
                     "gene","cell","atom","element","species","evolution",
                     "fertiliz","protein","DNA","RNA","energy","force"],
    "history":      ["war","battle","treaty","empire","king","queen","president",
                     "century","ancient","historical","revolution","dynasty",
                     "independence","founded","coloni"],
    "geography":    ["country","city","capital","continent","ocean","river",
                     "mountain","lake","island","region","located","border"],
    "entertainment":["movie","film","television","show","actor","actress","series",
                     "character","episode","song","album","music","band","award"],
    "sports":       ["nfl","nba","soccer","football","basketball","baseball",
                     "champion","league","team","player","tournament","olympic",
                     "score","win","season","coach","stadium"],
    "technology":   ["computer","software","internet","ai","algorithm","data",
                     "program","code","digital","network","device","platform"],
    "politics":     ["government","election","party","vote","senator","congress",
                     "parliament","law","policy","democrat","republican","minister"],
    "business":     ["company","corporation","ceo","revenue","market","stock",
                     "product","brand","industry","startup","invest","merger"],
}

def classify_domain(question: str, long_answer: str) -> str:
    combined = (question + " " + long_answer).lower()
    scores = {domain: sum(combined.count(kw) for kw in kws)
              for domain, kws in _DOMAIN_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ── Difficulty Estimation ─────────────────────────────────────────────────────

def estimate_difficulty(question: str, long_answer: str, short_answer: str) -> str:
    """
    Heuristic difficulty:
      easy   – has a short answer AND long answer ≤ 60 words
      medium – long answer 60–150 words, or no short answer
      hard   – long answer > 150 words, or complex multi-clause question
    """
    q_words    = len(question.split())
    ans_words  = len(long_answer.split()) if long_answer else 0
    has_short  = bool(short_answer and short_answer.strip())
    clauses    = question.count(",") + question.count(";") + question.count(" and ")

    if has_short and ans_words <= 60 and clauses <= 1:
        return "easy"
    elif ans_words > 150 or clauses > 2 or q_words > 20:
        return "hard"
    else:
        return "medium"


# ── Text Chunking ─────────────────────────────────────────────────────────────

# Very simple word-based tokenizer (no external tokenizer needed)
def _word_count(text: str) -> int:
    return len(text.split())

def _split_sentences(text: str) -> List[str]:
    """Split on sentence boundaries (.!?) while keeping the delimiter."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def chunk_text(text: str,
               max_tokens: int = None,
               overlap_tokens: int = None,
               min_tokens: int = None) -> List[Tuple[str, int]]:
    """
    Hybrid chunking:
      1. Split into sentences.
      2. Greedily pack sentences into windows ≤ max_tokens.
      3. Add overlap_tokens-word overlap between consecutive windows.

    Returns list of (chunk_text, word_count) tuples.
    """
    max_t = max_tokens    or CONFIG.chunking.max_chunk_tokens
    ovlp  = overlap_tokens or CONFIG.chunking.overlap_tokens
    min_t = min_tokens    or CONFIG.chunking.min_chunk_tokens

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[Tuple[str, int]] = []
    current_words: List[str] = []

    for sent in sentences:
        sent_words = sent.split()
        if _word_count(" ".join(current_words)) + len(sent_words) > max_t:
            # flush current window
            chunk = " ".join(current_words)
            if _word_count(chunk) >= min_t:
                chunks.append((chunk, _word_count(chunk)))
            # start new window with overlap
            current_words = current_words[-ovlp:] if ovlp else []
        current_words.extend(sent_words)

    # flush remainder
    if current_words:
        chunk = " ".join(current_words)
        if _word_count(chunk) >= min_t:
            chunks.append((chunk, _word_count(chunk)))

    return chunks


# ── Record Enrichment ─────────────────────────────────────────────────────────

def _make_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def enrich_record(raw: RawRecord) -> EnrichedRecord:
    """Add all metadata fields to a raw record."""
    q   = clean_text(raw.question)
    la  = clean_text(raw.long_answer)
    sa  = clean_text(raw.short_answer)
    return EnrichedRecord(
        record_id     = _make_id(q),
        row_id        = raw.row_id,
        question      = q,
        long_answer   = la,
        short_answer  = sa,
        has_short     = bool(sa),
        has_long      = bool(la),
        question_type = classify_question_type(q),
        domain        = classify_domain(q, la),
        difficulty    = estimate_difficulty(q, la, sa),
        answer_length = _word_count(la),
        language      = detect_language(q + " " + la),
    )


# ── Chunk Generation ──────────────────────────────────────────────────────────

def record_to_chunks(rec: EnrichedRecord) -> List[DocumentChunk]:
    """Convert one enriched record into one or more DocumentChunks."""
    chunks: List[DocumentChunk] = []
    base_meta = {
        k: v for k, v in asdict(rec).items()
        if k not in ("long_answer", "short_answer")
    }

    # ① Short-answer chunk (if present) — compact, high-confidence
    if rec.has_short:
        text = f"Q: {rec.question}\nA: {rec.short_answer}"
        chunks.append(DocumentChunk(
            chunk_id    = _make_id(text),
            record_id   = rec.record_id,
            row_id      = rec.row_id,
            question    = rec.question,
            text        = text,
            chunk_index = 0,
            chunk_type  = "short_answer",
            metadata    = {**base_meta, "chunk_type": "short_answer"},
            token_count = _word_count(text),
        ))

    # ② Long-answer chunks
    if rec.has_long:
        windows = chunk_text(rec.long_answer)
        for idx, (chunk_text_str, wc) in enumerate(windows):
            # Prepend question for richer context
            enriched_text = f"Q: {rec.question}\nContext: {chunk_text_str}"
            chunks.append(DocumentChunk(
                chunk_id    = _make_id(enriched_text + str(idx)),
                record_id   = rec.record_id,
                row_id      = rec.row_id,
                question    = rec.question,
                text        = enriched_text,
                chunk_index = idx + 1,
                chunk_type  = "long_answer_chunk",
                metadata    = {**base_meta, "chunk_type": "long_answer_chunk",
                               "window_index": idx},
                token_count = wc,
            ))

    # ③ Fallback: if no answers, index the question alone
    if not chunks:
        text = f"Q: {rec.question}"
        chunks.append(DocumentChunk(
            chunk_id    = _make_id(text),
            record_id   = rec.record_id,
            row_id      = rec.row_id,
            question    = rec.question,
            text        = text,
            chunk_index = 0,
            chunk_type  = "question_only",
            metadata    = {**base_meta, "chunk_type": "question_only"},
            token_count = _word_count(text),
        ))

    return chunks


# ── Main Dataset Loader ───────────────────────────────────────────────────────

class NaturalQuestionsProcessor:
    """
    End-to-end processor for the Natural Questions CSV dataset.

    Usage
    -----
    processor = NaturalQuestionsProcessor("Book1.csv")
    chunks    = processor.process(max_rows=1000)
    df_stats  = processor.statistics()
    """

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self._raw_df:       Optional[pd.DataFrame]    = None
        self._enriched:     List[EnrichedRecord]       = []
        self._chunks:       List[DocumentChunk]        = []

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    def load(self, max_rows: Optional[int] = None) -> "NaturalQuestionsProcessor":
        logger.info(f"Loading dataset from {self.csv_path} …")
        df = pd.read_csv(self.csv_path, usecols=["question","long_answers","short_answers"])
        df.columns = ["question", "long_answer", "short_answer"]
        df = df.dropna(subset=["question"])  # drop rows with no question
        df["long_answer"]  = df["long_answer"].fillna("")
        df["short_answer"] = df["short_answer"].fillna("")
        if max_rows:
            df = df.head(max_rows)
        df = df.reset_index(drop=True)
        self._raw_df = df
        logger.info(f"Loaded {len(df):,} rows.")
        return self

    # ── Step 2: Enrich ───────────────────────────────────────────────────────
    def enrich(self) -> "NaturalQuestionsProcessor":
        if self._raw_df is None:
            raise RuntimeError("Call .load() first.")
        logger.info("Enriching records (type, domain, difficulty, language) …")
        self._enriched = []
        for idx, row in self._raw_df.iterrows():
            raw = RawRecord(
                row_id       = int(idx),
                question     = row["question"],
                long_answer  = row["long_answer"],
                short_answer = row["short_answer"],
            )
            self._enriched.append(enrich_record(raw))
        logger.info(f"Enriched {len(self._enriched):,} records.")
        return self

    # ── Step 3: Chunk ────────────────────────────────────────────────────────
    def chunk(self) -> "NaturalQuestionsProcessor":
        if not self._enriched:
            raise RuntimeError("Call .enrich() first.")
        logger.info("Chunking documents …")
        self._chunks = []
        for rec in self._enriched:
            self._chunks.extend(record_to_chunks(rec))
        logger.info(f"Generated {len(self._chunks):,} chunks from "
                    f"{len(self._enriched):,} records.")
        return self

    # ── Full pipeline ────────────────────────────────────────────────────────
    def process(self, max_rows: Optional[int] = None) -> List[DocumentChunk]:
        return self.load(max_rows).enrich().chunk()._chunks

    # ── Statistics ───────────────────────────────────────────────────────────
    def statistics(self) -> pd.DataFrame:
        if not self._enriched:
            raise RuntimeError("No enriched records yet.")
        rows = [asdict(r) for r in self._enriched]
        df   = pd.DataFrame(rows)
        print("\n" + "="*60)
        print("  DATASET STATISTICS")
        print("="*60)
        print(f"  Total records       : {len(df):,}")
        print(f"  Has short answer    : {df['has_short'].sum():,} "
              f"({df['has_short'].mean()*100:.1f}%)")
        print(f"  Has long  answer    : {df['has_long'].sum():,} "
              f"({df['has_long'].mean()*100:.1f}%)")
        print(f"  Avg answer length   : {df['answer_length'].mean():.1f} words")
        print(f"\n  Question Types:\n{df['question_type'].value_counts().to_string()}")
        print(f"\n  Domains:\n{df['domain'].value_counts().to_string()}")
        print(f"\n  Difficulty:\n{df['difficulty'].value_counts().to_string()}")
        print(f"\n  Languages:\n{df['language'].value_counts().to_string()}")
        print("="*60)
        if self._chunks:
            chunk_df = pd.DataFrame([{"type": c.chunk_type,
                                       "tokens": c.token_count}
                                      for c in self._chunks])
            print(f"\n  Total chunks        : {len(chunk_df):,}")
            print(f"  Chunk types:\n{chunk_df['type'].value_counts().to_string()}")
            print(f"  Avg chunk tokens    : {chunk_df['tokens'].mean():.1f}")
            print("="*60)
        return df

    @property
    def enriched(self) -> List[EnrichedRecord]:
        return self._enriched

    @property
    def chunks(self) -> List[DocumentChunk]:
        return self._chunks
