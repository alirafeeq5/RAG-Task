"""
Evaluation Module
=================
Implements standard IR + QA evaluation metrics:
  • Precision@K, Recall@K
  • Mean Reciprocal Rank (MRR)
  • Normalised Discounted Cumulative Gain (NDCG)
  • Exact Match (EM)
  • F1 token overlap
  • BERTScore approximation (token-level cosine similarity)
  • End-to-end RAG pipeline evaluation
"""

import re
import math
import logging
import collections
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Token-level helpers
# ──────────────────────────────────────────────────────────────────────────────

_STOP = {"the", "a", "an", "is", "are", "was", "were", "in", "of", "and",
         "to", "it", "that", "this", "on", "at", "by", "with", "be"}

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

def _normalize(text: str) -> str:
    """Lowercase, remove articles, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ──────────────────────────────────────────────────────────────────────────────
#  Retrieval Metrics
# ──────────────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    top_k = retrieved[:k]
    hits  = sum(1 for d in top_k if d in relevant)
    return hits / max(k, 1)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of all relevant docs found in top-k."""
    top_k = retrieved[:k]
    hits  = sum(1 for d in top_k if d in relevant)
    return hits / max(len(relevant), 1)


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """Average precision across all relevant positions."""
    hits, running_prec = 0, 0.0
    for i, doc in enumerate(retrieved, 1):
        if doc in relevant:
            hits += 1
            running_prec += hits / i
    return running_prec / max(len(relevant), 1)


def mean_reciprocal_rank(retrieved_lists: List[List[str]],
                         relevant_sets:   List[Set[str]]) -> float:
    """
    MRR over a batch of queries.
    For each query, finds the rank of the first relevant document.
    """
    rr_sum = 0.0
    for retrieved, relevant in zip(retrieved_lists, relevant_sets):
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant:
                rr_sum += 1.0 / rank
                break
    return rr_sum / max(len(retrieved_lists), 1)


def dcg(scores: List[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(scores))

def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """NDCG@K: measures ranking quality weighted by position."""
    top_k    = retrieved[:k]
    gains    = [1.0 if d in relevant else 0.0 for d in top_k]
    ideal    = sorted(gains, reverse=True)
    dcg_val  = dcg(gains)
    idcg_val = dcg(ideal)
    return dcg_val / max(idcg_val, 1e-10)


# ──────────────────────────────────────────────────────────────────────────────
#  Answer Quality Metrics
# ──────────────────────────────────────────────────────────────────────────────

def exact_match(prediction: str, ground_truth: str) -> float:
    """Binary: 1 if normalised strings are identical, else 0."""
    return float(_normalize(prediction) == _normalize(ground_truth))


def token_f1(prediction: str, ground_truth: str) -> float:
    """
    Token-overlap F1 between prediction and ground truth.
    Standard metric for SQuAD-style evaluation.
    """
    pred_tokens = [t for t in _tokenize(_normalize(prediction)) if t not in _STOP]
    true_tokens = [t for t in _tokenize(_normalize(ground_truth)) if t not in _STOP]

    if not pred_tokens or not true_tokens:
        return float(pred_tokens == true_tokens)

    pred_counter = collections.Counter(pred_tokens)
    true_counter = collections.Counter(true_tokens)

    common = sum((pred_counter & true_counter).values())
    if common == 0:
        return 0.0

    precision = common / sum(pred_counter.values())
    recall    = common / sum(true_counter.values())
    return 2 * precision * recall / (precision + recall)


def semantic_similarity_approx(pred_vec: np.ndarray,
                                true_vec: np.ndarray) -> float:
    """
    Approximate BERTScore: cosine similarity between embedding vectors.
    (Full BERTScore needs transformers; this is a computationally cheap proxy.)
    """
    p = pred_vec.flatten()
    t = true_vec.flatten()
    np_  = np.linalg.norm(p)
    nt_  = np.linalg.norm(t)
    if np_ < 1e-10 or nt_ < 1e-10:
        return 0.0
    return float(np.dot(p, t) / (np_ * nt_))


# ──────────────────────────────────────────────────────────────────────────────
#  Full RAG Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalSample:
    query:          str
    ground_truth:   str                  # gold answer
    relevant_chunk_ids: List[str] = field(default_factory=list)  # gold doc ids


@dataclass
class EvalResult:
    query:        str
    prediction:   str
    ground_truth: str
    exact_match:  float
    token_f1:     float
    precision_1:  float
    recall_5:     float
    ndcg_5:       float
    mrr:          float
    latency_ms:   float


class RAGEvaluator:
    """
    Evaluates an end-to-end RAG pipeline on a set of labelled samples.

    Usage
    -----
    evaluator = RAGEvaluator(rag_pipeline)
    results   = evaluator.evaluate(samples, k=5)
    report    = evaluator.report(results)
    """

    def __init__(self, rag_pipeline):
        self._pipeline = rag_pipeline

    def evaluate(self,
                 samples:  List[EvalSample],
                 k:        int  = 5,
                 verbose:  bool = False) -> List[EvalResult]:
        results = []
        for i, sample in enumerate(samples):
            import time
            t0 = time.time()

            # Run pipeline
            response = self._pipeline.query(
                sample.query,
                top_k            = k,
                use_expansion    = True,
                use_rerank       = True,
                use_conversation = False,
            )
            prediction     = response.get("answer", "")
            retrieved_ids  = [r["chunk_id"]
                              for r in response.get("retrieved", [])]
            elapsed        = (time.time() - t0) * 1000

            # Retrieval metrics
            relevant_set = set(sample.relevant_chunk_ids)
            p1   = precision_at_k(retrieved_ids, relevant_set, 1)
            r5   = recall_at_k(retrieved_ids, relevant_set, k)
            nd5  = ndcg_at_k(retrieved_ids, relevant_set, k)
            mrr_ = mean_reciprocal_rank([retrieved_ids], [relevant_set])

            # Answer metrics
            em  = exact_match(prediction, sample.ground_truth)
            f1  = token_f1(prediction, sample.ground_truth)

            result = EvalResult(
                query        = sample.query,
                prediction   = prediction,
                ground_truth = sample.ground_truth,
                exact_match  = em,
                token_f1     = f1,
                precision_1  = p1,
                recall_5     = r5,
                ndcg_5       = nd5,
                mrr          = mrr_,
                latency_ms   = elapsed,
            )
            results.append(result)

            if verbose:
                logger.info(
                    f"[{i+1}/{len(samples)}] Q: {sample.query[:60]} "
                    f"| EM={em:.2f} F1={f1:.2f} NDCG={nd5:.2f}"
                )

        return results

    @staticmethod
    def report(results: List[EvalResult]) -> Dict:
        if not results:
            return {}
        n = len(results)
        avg = lambda key: sum(getattr(r, key) for r in results) / n

        report_ = {
            "n_samples":       n,
            "exact_match":     round(avg("exact_match"), 3),
            "token_f1":        round(avg("token_f1"), 3),
            "precision@1":     round(avg("precision_1"), 3),
            "recall@5":        round(avg("recall_5"), 3),
            "ndcg@5":          round(avg("ndcg_5"), 3),
            "mrr":             round(avg("mrr"), 3),
            "avg_latency_ms":  round(avg("latency_ms"), 1),
        }

        print("\n" + "="*55)
        print("  RAG EVALUATION REPORT")
        print("="*55)
        for k, v in report_.items():
            print(f"  {k:<22}: {v}")
        print("="*55)
        return report_
