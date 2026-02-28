"""
Phase 2 – Query Processing & Enhancement
=========================================
Covers:
  • Query preprocessing & normalisation
  • Multi-turn conversation context management
  • Query expansion (synonym injection, sub-question decomposition)
  • Relevance re-ranking (BM25-style + MMR diversity)
"""

import re
import math
import time
import logging
import collections
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any

import numpy as np

from config import CONFIG

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  1. Query Normalisation
# ──────────────────────────────────────────────────────────────────────────────

_FILLER_PHRASES = re.compile(
    r"\b(please|can you|could you|would you|tell me|i want to know|"
    r"what is the answer to|give me|i need|help me|just)\b",
    re.IGNORECASE,
)
_WS_MULTI  = re.compile(r"\s{2,}")
_PUNCT_END = re.compile(r"[?.!,;:]+$")

def preprocess_query(query: str) -> str:
    """
    Normalise a raw user query:
      1. Strip leading/trailing whitespace.
      2. Remove filler phrases that add noise.
      3. Collapse whitespace.
      4. Ensure ends without trailing punctuation for embedding consistency.
    """
    q = query.strip()
    q = _FILLER_PHRASES.sub(" ", q)
    q = _WS_MULTI.sub(" ", q).strip()
    # Lowercase only if entirely uppercase (accidental CAPS LOCK)
    if q == q.upper() and len(q) > 5:
        q = q.lower()
    return q


# ──────────────────────────────────────────────────────────────────────────────
#  2. Multi-Turn Conversation Context
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    role:      str   # "user" | "assistant"
    content:   str
    timestamp: float = field(default_factory=time.time)


class ConversationContext:
    """
    Manages a sliding window of conversation turns.
    Provides a contextualised query that incorporates recent history.
    """

    def __init__(self, window: int = None):
        self._turns: List[Turn] = []
        self._window = window or CONFIG.retrieval.context_window

    def add_turn(self, role: str, content: str) -> None:
        self._turns.append(Turn(role=role, content=content))

    def get_contextual_query(self, new_query: str) -> str:
        """
        Combine the last N assistant responses with the new user query so
        the embedding captures conversational co-reference.

        Example
        -------
        Previous assistant: "Tom Brady plays for the New England Patriots."
        New user query:     "How many Super Bowls did he win?"
        Contextual:         "Tom Brady plays for the New England Patriots.
                             How many Super Bowls did he win?"
        """
        recent_assistant = [
            t.content for t in self._turns[-self._window * 2:]
            if t.role == "assistant"
        ][-self._window:]

        if not recent_assistant:
            return new_query

        context_snippet = " ".join(recent_assistant[-2:])  # last 2 assistant turns
        # Keep snippet short to not overwhelm the query
        if len(context_snippet.split()) > 60:
            context_snippet = " ".join(context_snippet.split()[:60])

        return f"{context_snippet} {new_query}"

    def get_history(self) -> List[dict]:
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turns(self) -> List[Turn]:
        return list(self._turns)


# ──────────────────────────────────────────────────────────────────────────────
#  3. Query Expansion
# ──────────────────────────────────────────────────────────────────────────────

# Mini synonym/hypernym table for common question tokens
_SYNONYMS: Dict[str, List[str]] = {
    "who":       ["person", "individual", "author", "founder", "creator"],
    "when":      ["year", "date", "time", "period", "era"],
    "where":     ["location", "place", "country", "city", "region"],
    "what":      ["definition", "meaning", "description", "type"],
    "how":       ["method", "process", "way", "steps", "procedure"],
    "why":       ["reason", "cause", "explanation", "purpose"],
    "first":     ["inaugural", "initial", "earliest", "original"],
    "largest":   ["biggest", "greatest", "most extensive"],
    "smallest":  ["tiniest", "least", "minimum"],
    "capital":   ["capital city", "seat of government"],
    "president": ["head of state", "leader", "chief executive"],
    "founder":   ["creator", "originator", "established by"],
    "invented":  ["created", "developed", "designed", "built"],
    "discover":  ["found", "identified", "identified"],
}

def expand_query(query: str, top_n: int = 3) -> List[str]:
    """
    Generate query variants by:
      a) Synonym substitution for key terms.
      b) Keyword-only variant (strip question words).
      c) Declarative form (convert question to statement).

    Returns list of expanded queries (including the original).
    """
    words   = query.lower().split()
    variants: List[str] = [query]

    # (a) Synonym injection: append synonyms as extra context terms
    extra_terms = []
    for w in words:
        syns = _SYNONYMS.get(w.rstrip("?!.,"), [])
        extra_terms.extend(syns[:2])
    if extra_terms:
        variants.append(query + " " + " ".join(set(extra_terms[:top_n])))

    # (b) Keyword-only: drop WH-words and auxiliary verbs
    stop_words = {"what","who","when","where","how","why","which","is","are",
                  "was","were","did","does","do","the","a","an","of","in","on"}
    keywords = [w for w in words if w.rstrip("?!.,") not in stop_words]
    if keywords and len(keywords) < len(words):
        variants.append(" ".join(keywords))

    # (c) Declarative form: "Who invented X?" → "X was invented by"
    decl = _to_declarative(query)
    if decl and decl != query:
        variants.append(decl)

    return list(dict.fromkeys(variants))[:top_n + 1]   # dedup, limit


def _to_declarative(question: str) -> str:
    """Heuristically convert a question into a declarative phrase."""
    q = question.strip().rstrip("?")
    # "Who invented X" → "X was invented"
    m = re.match(r"who\s+(invented|created|founded|discovered)\s+(.+)", q, re.I)
    if m:
        return f"{m.group(2).strip()} was {m.group(1)}"
    # "When was X born" → "X birth year"
    m = re.match(r"when\s+was\s+(.+?)\s+(born|founded|established|built)", q, re.I)
    if m:
        return f"{m.group(1).strip()} {m.group(2)}"
    # "What is the capital of X" → "capital of X"
    m = re.match(r"what\s+is\s+the?\s+(.+)", q, re.I)
    if m:
        return m.group(1).strip()
    return ""


# ──────────────────────────────────────────────────────────────────────────────
#  4. Relevance Scoring & Re-Ranking
# ──────────────────────────────────────────────────────────────────────────────

class BM25Scorer:
    """
    BM25 re-ranker that can re-score retrieved chunks against the query.
    Used as a lightweight complement to dense retrieval.

    Parameters
    ----------
    k1 : float – term frequency saturation parameter (default 1.5)
    b  : float – field-length normalisation parameter (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0
        self._corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, corpus: List[str]) -> "BM25Scorer":
        tokenised    = [self._tokenize(doc) for doc in corpus]
        self._corpus = tokenised
        self._avgdl  = sum(len(d) for d in tokenised) / max(len(tokenised), 1)
        N = len(tokenised)
        df: Dict[str, int] = collections.defaultdict(int)
        for doc in tokenised:
            for term in set(doc):
                df[term] += 1
        self._idf = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }
        return self

    def score(self, query: str, doc: str) -> float:
        q_terms = self._tokenize(query)
        d_terms = self._tokenize(doc)
        dl      = len(d_terms)
        tf_map  = collections.Counter(d_terms)
        score   = 0.0
        for term in q_terms:
            idf = self._idf.get(term, 0.0)
            tf  = tf_map.get(term, 0)
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1))
            score += idf * num / max(den, 1e-10)
        return score


def mmr_rerank(
    query_vec:   np.ndarray,
    candidates:  List[dict],
    embed_fn,
    top_k:       int  = 5,
    lam:         float = 0.7,
) -> List[dict]:
    """
    Maximal Marginal Relevance re-ranking.
    Balances relevance (λ) with diversity (1-λ) to avoid redundant results.

    Parameters
    ----------
    query_vec   : (1, dim) query embedding
    candidates  : list of retrieval dicts (must have 'text' key)
    embed_fn    : callable(texts) → np.ndarray
    top_k       : desired output size
    lam         : relevance weight (0=max diversity, 1=max relevance)
    """
    if not candidates:
        return []
    k = min(top_k, len(candidates))

    texts = [c["text"] for c in candidates]
    vecs  = embed_fn(texts)                         # (N, dim)

    q  = query_vec.flatten()
    q_norm = np.linalg.norm(q)
    if q_norm > 1e-10:
        q = q / q_norm

    # Relevance scores: cosine(query, doc_i)
    norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms  = np.where(norms < 1e-10, 1e-10, norms)
    normed = vecs / norms
    rel    = (normed @ q).astype(np.float32)

    selected_idx: List[int] = []
    remaining    = list(range(len(candidates)))

    while len(selected_idx) < k and remaining:
        if not selected_idx:
            # First pick: highest relevance
            best = max(remaining, key=lambda i: rel[i])
        else:
            # Subsequent picks: balance relevance vs. similarity to already-selected
            sel_vecs = normed[selected_idx]    # (S, dim)
            best_score = -np.inf
            best = remaining[0]
            for i in remaining:
                sim_to_sel = float(np.max(normed[i] @ sel_vecs.T))
                mmr_score  = lam * rel[i] - (1 - lam) * sim_to_sel
                if mmr_score > best_score:
                    best_score = mmr_score
                    best = i
        selected_idx.append(best)
        remaining.remove(best)

    return [candidates[i] for i in selected_idx]


def hybrid_rerank(
    query:      str,
    candidates: List[dict],
    embed_fn,
    query_vec:  Optional[np.ndarray] = None,
    top_k:      int   = 5,
    bm25_weight: float = 0.3,
    dense_weight: float = 0.7,
) -> List[dict]:
    """
    Combine BM25 sparse scores with dense cosine scores via weighted sum.
    """
    if not candidates:
        return []

    texts  = [c["text"] for c in candidates]
    bm25   = BM25Scorer()
    bm25.fit(texts)

    # BM25 scores
    bm25_scores = np.array([bm25.score(query, t) for t in texts])
    max_bm25 = bm25_scores.max()
    if max_bm25 > 0:
        bm25_scores = bm25_scores / max_bm25   # normalise to [0,1]

    # Dense scores (already in each candidate)
    dense_scores = np.array([c.get("score", 0.0) for c in candidates])
    min_d, max_d = dense_scores.min(), dense_scores.max()
    if max_d > min_d:
        dense_scores = (dense_scores - min_d) / (max_d - min_d)

    # Combined
    combined = bm25_weight * bm25_scores + dense_weight * dense_scores
    ranked   = sorted(range(len(candidates)), key=lambda i: -combined[i])

    result = []
    for rank, i in enumerate(ranked[:top_k]):
        c = dict(candidates[i])
        c["score"]        = float(combined[i])
        c["dense_score"]  = float(dense_scores[i])
        c["bm25_score"]   = float(bm25_scores[i])
        c["rank"]         = rank
        result.append(c)
    return result


# ──────────────────────────────────────────────────────────────────────────────
#  5. Query Processor (orchestrates all of the above)
# ──────────────────────────────────────────────────────────────────────────────

class QueryProcessor:
    """
    End-to-end query processing pipeline:
      preprocess → expand → embed → retrieve → rerank
    """

    def __init__(self, document_index, embedding_engine):
        self._index   = document_index
        self._engine  = embedding_engine
        self._context = ConversationContext()

    def process(self,
                raw_query:         str,
                top_k:             int   = None,
                score_threshold:   float = None,
                use_expansion:     bool  = True,
                use_rerank:        bool  = True,
                use_conversation:  bool  = True) -> dict:
        """
        Full query processing pipeline.

        Returns
        -------
        {
          "original_query":    str,
          "processed_query":   str,
          "expanded_queries":  List[str],
          "results":           List[dict],
          "context_used":      bool,
        }
        """
        t0 = time.time()
        k  = top_k or CONFIG.retrieval.top_k
        thr = score_threshold if score_threshold is not None \
              else CONFIG.retrieval.score_threshold

        # Step 1 – preprocess
        clean_q = preprocess_query(raw_query)

        # Step 2 – contextualise
        ctx_q = (self._context.get_contextual_query(clean_q)
                 if use_conversation else clean_q)
        ctx_used = ctx_q != clean_q

        # Step 3 – expand
        expanded = expand_query(ctx_q, top_n=3) if use_expansion else [ctx_q]

        # Step 4 – retrieve (merge results from all expanded queries)
        seen: Dict[str, dict] = {}
        for eq in expanded:
            for r in self._index.search(eq, top_k=k * 2,
                                         score_threshold=thr * 0.7):
                cid = r["chunk_id"]
                if cid not in seen or r["score"] > seen[cid]["score"]:
                    seen[cid] = r

        candidates = sorted(seen.values(), key=lambda x: -x["score"])

        # Step 5 – re-rank
        if use_rerank and len(candidates) > 1:
            q_vec = self._engine.embed_query(ctx_q)
            candidates = hybrid_rerank(
                ctx_q, candidates,
                embed_fn     = self._engine.embed_texts,
                query_vec    = q_vec,
                top_k        = k,
                bm25_weight  = 0.3,
                dense_weight = 0.7,
            )
        else:
            candidates = candidates[:k]

        elapsed = time.time() - t0
        logger.debug(f"Query processed in {elapsed*1000:.1f}ms – "
                     f"{len(candidates)} results")

        return {
            "original_query":   raw_query,
            "processed_query":  ctx_q,
            "expanded_queries": expanded,
            "results":          candidates,
            "context_used":     ctx_used,
            "elapsed_ms":       round(elapsed * 1000, 1),
        }

    def add_to_history(self, role: str, content: str) -> None:
        self._context.add_turn(role, content)

    def reset_context(self) -> None:
        self._context.clear()

    @property
    def conversation_history(self) -> List[dict]:
        return self._context.get_history()
