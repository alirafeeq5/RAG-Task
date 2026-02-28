"""
Phase 2 – Response Generation
===============================
Covers:
  • Groq LLM API integration (async + sync)
  • Context-aware prompt engineering
  • Response quality validation & filtering
  • Fallback mechanisms for low-confidence answers
  • Caching layer (in-memory LRU + optional disk)
"""

import hashlib
import json
import logging
import re
import time
import collections
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from config import CONFIG

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Prompt Engineering
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable, accurate, and concise question-answering assistant.
You answer questions using ONLY the provided context passages.

Guidelines:
- Provide direct, factual answers grounded in the context.
- If the context does not contain sufficient information, say so clearly.
- Keep answers concise (1-3 sentences for factual questions, up to a paragraph for complex ones).
- Do NOT hallucinate or add information not present in the context.
- Cite the most relevant passage when appropriate.
- Maintain a neutral, informative tone."""

def build_rag_prompt(
    query:    str,
    contexts: List[Dict],
    history:  Optional[List[Dict]] = None,
    language: str = "en",
) -> List[Dict]:
    """
    Construct the messages payload for the Groq/OpenAI chat API.

    Parameters
    ----------
    query    : the user's question
    contexts : list of retrieved chunks [{text, score, metadata}, …]
    history  : prior conversation turns [{role, content}, …]
    language : detected language of the query (passed as hint)

    Returns
    -------
    List of message dicts for the API call.
    """
    # Format context passages
    ctx_blocks = []
    for i, ctx in enumerate(contexts[:5], 1):           # cap at 5 passages
        meta  = ctx.get("metadata", {})
        score = ctx.get("score", 0.0)
        dom   = meta.get("domain", "general")
        diff  = meta.get("difficulty", "unknown")
        ctx_blocks.append(
            f"[Passage {i} | domain={dom} | difficulty={diff} | relevance={score:.2f}]\n"
            f"{ctx['text']}"
        )

    context_str = "\n\n".join(ctx_blocks) if ctx_blocks else "No relevant context found."

    # Language hint
    lang_hint = ""
    if language != "en":
        lang_hint = (f"\n(The user's question appears to be in language code: {language}. "
                     "Respond in the same language if possible.)")

    user_content = (
        f"Context passages:\n{context_str}\n\n"
        f"Question: {query}{lang_hint}\n\n"
        "Answer based strictly on the context above:"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Include limited conversation history
    if history:
        for turn in history[-4:]:          # last 2 Q&A pairs
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_content})
    return messages


# ──────────────────────────────────────────────────────────────────────────────
#  Response Quality Validation
# ──────────────────────────────────────────────────────────────────────────────

_UNCERTAINTY_PHRASES = re.compile(
    r"\b(i (don'?t|do not) (know|have)|"
    r"no information|not (mentioned|found|available|provided)|"
    r"cannot (determine|find|answer)|"
    r"insufficient (context|information)|"
    r"context does not (contain|include|mention))\b",
    re.IGNORECASE,
)

_HALLUCINATION_MARKERS = re.compile(
    r"\b(as of my (knowledge|training)|"
    r"based on my (training|knowledge)|"
    r"i (was trained|know from)|"
    r"according to (my|general) knowledge)\b",
    re.IGNORECASE,
)

def validate_response(answer: str, query: str, contexts: List[Dict]) -> Dict:
    """
    Heuristic quality checks on the generated answer.

    Returns
    -------
    {
      "is_valid":      bool,
      "confidence":    float,   # 0-1
      "issues":        List[str],
      "is_uncertain":  bool,
      "hallucinated":  bool,
    }
    """
    issues: List[str] = []
    confidence = 1.0

    # (a) Minimum length
    if len(answer.split()) < CONFIG.llm.min_answer_len:
        issues.append("answer_too_short")
        confidence -= 0.3

    # (b) Explicit uncertainty expressed by model
    is_uncertain = bool(_UNCERTAINTY_PHRASES.search(answer))
    if is_uncertain:
        issues.append("model_expressed_uncertainty")
        confidence -= 0.4

    # (c) Hallucination markers (model leaking its training knowledge)
    hallucinated = bool(_HALLUCINATION_MARKERS.search(answer))
    if hallucinated:
        issues.append("potential_hallucination")
        confidence -= 0.5

    # (d) Overlap with context (simple token overlap sanity check)
    if contexts:
        ctx_tokens = set(
            re.findall(r"\b\w+\b",
                       " ".join(c["text"] for c in contexts).lower())
        )
        ans_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
        stop = {"the","a","an","is","are","was","were","in","of","and","to","it"}
        overlap = (ctx_tokens - stop) & (ans_tokens - stop)
        overlap_ratio = len(overlap) / max(len(ans_tokens - stop), 1)
        if overlap_ratio < 0.05:
            issues.append("low_context_overlap")
            confidence -= 0.2

    confidence = max(0.0, min(1.0, confidence))
    return {
        "is_valid":     confidence >= 0.4,
        "confidence":   round(confidence, 2),
        "issues":       issues,
        "is_uncertain": is_uncertain,
        "hallucinated": hallucinated,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Caching
# ──────────────────────────────────────────────────────────────────────────────

class ResponseCache:
    """
    Two-level cache:
      L1 – in-memory LRU dict (fast, bounded size)
      L2 – disk JSON cache (survives restarts if CACHE_DIR is persistent)
    """

    def __init__(self):
        cfg = CONFIG.cache
        self._enabled  = cfg.enabled
        self._max_size = cfg.max_size
        self._ttl      = cfg.ttl_seconds
        self._cache_dir = cfg.cache_dir
        # L1
        self._lru: collections.OrderedDict = collections.OrderedDict()
        # Stats
        self._hits   = 0
        self._misses = 0

    def _key(self, query: str, context_ids: List[str]) -> str:
        raw = query.lower().strip() + "|" + ",".join(sorted(context_ids))
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, query: str, context_ids: List[str]) -> Optional[str]:
        if not self._enabled:
            return None
        k = self._key(query, context_ids)

        # L1
        if k in self._lru:
            entry = self._lru[k]
            if time.time() - entry["ts"] < self._ttl:
                self._lru.move_to_end(k)
                self._hits += 1
                return entry["answer"]
            else:
                del self._lru[k]

        # L2 (disk)
        path = self._cache_dir / f"{k}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text())
                if time.time() - data["ts"] < self._ttl:
                    self._lru[k] = data
                    self._hits += 1
                    return data["answer"]
            except Exception:
                pass

        self._misses += 1
        return None

    def set(self, query: str, context_ids: List[str], answer: str) -> None:
        if not self._enabled:
            return
        k     = self._key(query, context_ids)
        entry = {"answer": answer, "ts": time.time()}
        # L1
        self._lru[k] = entry
        self._lru.move_to_end(k)
        if len(self._lru) > self._max_size:
            self._lru.popitem(last=False)
        # L2 (disk) – best-effort
        try:
            (self._cache_dir / f"{k}.json").write_text(json.dumps(entry))
        except Exception:
            pass

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits":      self._hits,
            "misses":    self._misses,
            "hit_rate":  round(self._hits / max(total, 1), 3),
            "l1_size":   len(self._lru),
        }


# ──────────────────────────────────────────────────────────────────────────────
#  LLM Client (Groq / OpenAI-compatible)
# ──────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Async-compatible HTTP client for Groq (or any OpenAI-compatible endpoint).

    When the network is unavailable, _mock_response() is used so the
    pipeline can be tested end-to-end without a real API key.
    """

    def __init__(self):
        self._cfg = CONFIG.llm
        self._cache = ResponseCache()
        self._total_calls  = 0
        self._failed_calls = 0

    # ── core call ────────────────────────────────────────────────────────────
    def call(self, messages: List[Dict], use_cache: bool = True) -> str:
        """
        Send messages to the LLM and return the assistant's text reply.
        Falls back gracefully if the API is unreachable.
        """
        self._total_calls += 1
        ctx_ids = []  # for cache key

        # Cache lookup
        if use_cache and messages:
            user_msg = next((m["content"] for m in reversed(messages)
                             if m["role"] == "user"), "")
            cached = self._cache.get(user_msg, ctx_ids)
            if cached is not None:
                logger.debug("Cache hit – returning cached answer.")
                return cached

        answer = self._call_api(messages)

        # Cache store
        if use_cache and messages:
            user_msg = next((m["content"] for m in reversed(messages)
                             if m["role"] == "user"), "")
            self._cache.set(user_msg, ctx_ids, answer)

        return answer

    def _call_api(self, messages: List[Dict]) -> str:
        """Attempt real API call; fall back to mock on any error."""
        try:
            import urllib.request
            import urllib.error

            payload = json.dumps({
                "model":       self._cfg.model,
                "max_tokens":  self._cfg.max_tokens,
                "temperature": self._cfg.temperature,
                "messages":    messages,
            }).encode()

            req = urllib.request.Request(
                url     = f"{self._cfg.base_url}/chat/completions",
                data    = payload,
                method  = "POST",
                headers = {
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {self._cfg.api_key}",
                },
            )
            with urllib.request.urlopen(req,
                                        timeout=self._cfg.timeout_seconds) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            self._failed_calls += 1
            logger.warning(f"LLM API call failed ({e}). Using mock response.")
            return self._mock_response(messages)

    def _mock_response(self, messages: List[Dict]) -> str:
        """
        Offline mock: extract the most relevant sentence from the
        provided context passages when the API is unavailable.
        """
        user_msg = next((m["content"] for m in reversed(messages)
                         if m["role"] == "user"), "")

        # Parse context from prompt
        ctx_match = re.search(
            r"Context passages:\n(.+?)\nQuestion:",
            user_msg, re.DOTALL
        )
        if not ctx_match:
            return CONFIG.llm.fallback_msg

        ctx_text = ctx_match.group(1)
        q_match  = re.search(r"Question:\s*(.+?)(?:\n|$)", user_msg)
        question = q_match.group(1).strip() if q_match else ""

        # Find best sentence by keyword overlap
        sentences = re.split(r"(?<=[.!?])\s+", ctx_text)
        q_words   = set(re.findall(r"\b\w+\b", question.lower()))
        stop      = {"the","a","an","is","are","was","were","in","of","and",
                     "to","it","what","who","when","where","how","why"}
        q_words  -= stop

        best_sent, best_score = "", 0
        for sent in sentences:
            if len(sent) < 20:
                continue
            s_words = set(re.findall(r"\b\w+\b", sent.lower()))
            overlap = len(q_words & s_words)
            if overlap > best_score:
                best_score = overlap
                best_sent  = sent

        if best_sent and best_score > 0:
            # Clean up passage header
            best_sent = re.sub(r"\[Passage \d+[^\]]*\]\n?", "", best_sent).strip()
            return best_sent[:500] if best_sent else CONFIG.llm.fallback_msg

        return CONFIG.llm.fallback_msg

    @property
    def stats(self) -> dict:
        return {
            "total_calls":  self._total_calls,
            "failed_calls": self._failed_calls,
            "success_rate": round(
                1 - self._failed_calls / max(self._total_calls, 1), 3),
            "cache":        self._cache.stats,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Response Generator (orchestrates prompt + LLM + validation + fallback)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    answer:          str
    is_valid:        bool
    confidence:      float
    quality_issues:  List[str]
    used_fallback:   bool
    source_chunks:   List[str]    # chunk_ids used
    latency_ms:      float


class ResponseGenerator:
    """
    Generates answers by combining retrieved context with an LLM.

    Usage
    -----
    gen = ResponseGenerator()
    result = gen.generate(query, retrieved_chunks, history)
    """

    def __init__(self):
        self._client = LLMClient()

    def generate(
        self,
        query:       str,
        contexts:    List[Dict],
        history:     Optional[List[Dict]] = None,
        language:    str = "en",
        use_cache:   bool = True,
    ) -> GenerationResult:

        t0 = time.time()

        # No context → immediate fallback
        if not contexts:
            return GenerationResult(
                answer         = CONFIG.llm.fallback_msg,
                is_valid       = False,
                confidence     = 0.0,
                quality_issues = ["no_context"],
                used_fallback  = True,
                source_chunks  = [],
                latency_ms     = 0.0,
            )

        # Build prompt
        messages = build_rag_prompt(query, contexts, history, language)

        # Call LLM
        answer = self._client.call(messages, use_cache=use_cache)

        # Validate
        validation = validate_response(answer, query, contexts)

        # Fallback if quality is too low
        used_fallback = False
        if not validation["is_valid"]:
            # Try once with a simpler prompt (extract the best short answer)
            fallback_answer = self._extract_direct_answer(query, contexts)
            if fallback_answer:
                answer        = fallback_answer
                used_fallback = True
                validation    = validate_response(answer, query, contexts)
            else:
                answer        = CONFIG.llm.fallback_msg
                used_fallback = True

        latency = (time.time() - t0) * 1000
        return GenerationResult(
            answer         = answer,
            is_valid       = validation["is_valid"],
            confidence     = validation["confidence"],
            quality_issues = validation["issues"],
            used_fallback  = used_fallback,
            source_chunks  = [c.get("chunk_id", "") for c in contexts],
            latency_ms     = round(latency, 1),
        )

    @staticmethod
    def _extract_direct_answer(query: str, contexts: List[Dict]) -> str:
        """
        Fallback: look for a 'short_answer' chunk; if found return it directly.
        """
        for ctx in contexts:
            if ctx.get("metadata", {}).get("chunk_type") == "short_answer":
                text = ctx["text"]
                # Strip "Q: … A: " prefix
                m = re.search(r"\bA:\s*(.+)", text)
                if m:
                    return m.group(1).strip()[:300]
        return ""

    @property
    def stats(self) -> dict:
        return self._client.stats
