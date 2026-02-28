"""
Test Suite – Multilingual RAG System
=====================================
Unit + integration tests for all major components.
Run:  python -m pytest tests/test_rag.py -v
  or: python tests/test_rag.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import re
import math
import json
import time
import logging
import unittest
import numpy as np
import tempfile
from pathlib import Path

logger = logging.getLogger("tests")


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_sample_csv(path: str, n: int = 30) -> None:
    """Write a tiny sample CSV for testing."""
    rows = [
        ("who invented the telephone", 
         "<P> Alexander Graham Bell is credited with inventing the telephone. "
         "He received the first patent for the telephone in 1876. </P>",
         "Alexander Graham Bell"),
        ("what is the capital of france",
         "<P> Paris is the capital and most populous city of France. It is "
         "situated on the Seine River, in northern France. </P>",
         "Paris"),
        ("when was the eiffel tower built",
         "<P> The Eiffel Tower was constructed from 1887 to 1889 as the entrance "
         "arch to the 1889 World's Fair in Paris. </P>",
         "1887 to 1889"),
        ("what is photosynthesis",
         "<P> Photosynthesis is the process by which plants use sunlight, water, "
         "and carbon dioxide to produce oxygen and energy in the form of glucose. </P>",
         "process by which plants convert sunlight to energy"),
        ("who wrote hamlet",
         "<P> Hamlet is a tragedy written by William Shakespeare, believed to have "
         "been written around 1600. </P>",
         "William Shakespeare"),
    ]
    import csv, io
    lines = ["question,long_answers,short_answers"]
    for q, la, sa in rows * (n // len(rows) + 1):
        lines.append(f'"{q}","{la}","{sa}"')
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines[:n+1]))


# ──────────────────────────────────────────────────────────────────────────────
#  Unit Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w")
        self.tmp.close()
        make_sample_csv(self.tmp.name, 20)

    def test_load(self):
        from core.data_processor import NaturalQuestionsProcessor
        p = NaturalQuestionsProcessor(self.tmp.name)
        p.load(max_rows=10)
        self.assertEqual(len(p._raw_df), 10)

    def test_enrich(self):
        from core.data_processor import NaturalQuestionsProcessor
        p = NaturalQuestionsProcessor(self.tmp.name)
        p.load(max_rows=5).enrich()
        self.assertEqual(len(p.enriched), 5)
        for rec in p.enriched:
            self.assertIn(rec.question_type,
                          ["what","who","when","where","how","why","which","is_are","other"])
            self.assertIn(rec.difficulty, ["easy","medium","hard"])
            self.assertIn(rec.domain, ["science","history","geography",
                                        "entertainment","sports","technology",
                                        "politics","business","general"])

    def test_chunk(self):
        from core.data_processor import NaturalQuestionsProcessor
        p = NaturalQuestionsProcessor(self.tmp.name)
        chunks = p.process(max_rows=5)
        self.assertGreater(len(chunks), 5)
        for c in chunks:
            self.assertIsNotNone(c.chunk_id)
            self.assertGreater(c.token_count, 0)
            self.assertIn(c.chunk_type,
                          ["short_answer", "long_answer_chunk", "question_only"])

    def test_clean_text(self):
        from core.data_processor import clean_text
        self.assertEqual(clean_text("<P> Hello World . </P>"), "Hello World.")
        self.assertEqual(clean_text("  extra   spaces  "), "extra spaces")

    def test_language_detection(self):
        from core.data_processor import detect_language
        self.assertEqual(detect_language("Hello world this is English"), "en")
        self.assertEqual(detect_language("مرحبا بالعالم"), "ar")

    def test_question_type(self):
        from core.data_processor import classify_question_type
        self.assertEqual(classify_question_type("Who is Einstein?"), "who")
        self.assertEqual(classify_question_type("When was Rome founded?"), "when")
        self.assertEqual(classify_question_type("What is DNA?"), "what")

    def test_chunk_overlap(self):
        from core.data_processor import chunk_text
        long_text = " ".join([f"sentence{i} is here." for i in range(100)])
        chunks = chunk_text(long_text, max_tokens=30, overlap_tokens=5)
        self.assertGreater(len(chunks), 1)
        # Verify overlap: last N words of chunk[i] should appear in chunk[i+1]
        for i in range(len(chunks)-1):
            words_a = chunks[i][0].split()
            words_b = chunks[i+1][0].split()
            # Some words from end of A should appear at start of B
            last_a = set(words_a[-10:])
            first_b = set(words_b[:15])
            self.assertTrue(len(last_a & first_b) > 0,
                            "Expected overlap between consecutive chunks")


class TestEmbedding(unittest.TestCase):

    def setUp(self):
        from core.embeddings import EmbeddingEngine
        self.engine = EmbeddingEngine(prefer_transformer=False)
        texts = [
            "Alexander Graham Bell invented the telephone.",
            "Paris is the capital of France.",
            "Shakespeare wrote Hamlet.",
            "Photosynthesis converts sunlight into energy.",
        ]
        self.engine.fit_corpus(texts)
        self.texts = texts

    def test_embed_texts_shape(self):
        vecs = self.engine.embed_texts(self.texts)
        self.assertEqual(vecs.shape[0], len(self.texts))
        self.assertGreater(vecs.shape[1], 0)

    def test_embed_query_shape(self):
        qvec = self.engine.embed_query("Who invented the telephone?")
        self.assertEqual(qvec.shape[0], 1)

    def test_cosine_similarity(self):
        vecs = self.engine.embed_texts(self.texts)
        # Semantically similar texts should have higher cosine sim
        sim_same = self.engine.cosine_similarity(vecs[0], vecs[0])
        self.assertAlmostEqual(sim_same, 1.0, places=3)

    def test_normalization(self):
        vecs = self.engine.embed_texts(self.texts[:2])
        for v in vecs:
            norm = float(np.linalg.norm(v))
            self.assertAlmostEqual(norm, 1.0, places=3,
                                   msg="Vectors should be L2-normalised")

    def test_cache(self):
        q = "Who invented the telephone?"
        self.engine.embed_query(q)
        self.engine.embed_query(q)    # second call should be cached
        stats = self.engine.cache_stats()
        self.assertGreater(stats["hits"], 0)


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        from core.vector_store import NumpyVectorStore
        self.dim = 50
        self.store = NumpyVectorStore(dim=self.dim)
        # Add random normalised vectors
        np.random.seed(42)
        vecs = np.random.randn(20, self.dim).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        self.vecs = vecs / norms
        self.ids  = [f"chunk_{i}" for i in range(20)]
        self.store.add(self.vecs, self.ids)

    def test_size(self):
        self.assertEqual(self.store.size, 20)

    def test_search_returns_top_k(self):
        q = self.vecs[0]
        results = self.store.search(q, top_k=5)
        self.assertEqual(len(results), 5)

    def test_top_result_is_self(self):
        """Querying with a stored vector should return that vector first."""
        q = self.vecs[3]
        results = self.store.search(q, top_k=3)
        self.assertEqual(results[0].chunk_id, self.ids[3])

    def test_scores_descending(self):
        q = self.vecs[0]
        results = self.store.search(q, top_k=10)
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from config import INDEX_DIR
            orig_path = INDEX_DIR
            # Temporarily redirect save path
            from core.vector_store import NumpyVectorStore
            s2 = NumpyVectorStore(dim=self.dim)
            s2._index_path = Path(tmpdir) / "idx.npz"
            s2._meta_path  = Path(tmpdir) / "meta.json"
            s2.add(self.vecs, self.ids)
            s2.save()
            s3 = NumpyVectorStore(dim=self.dim)
            s3._index_path = s2._index_path
            s3._meta_path  = s2._meta_path
            s3.load()
            self.assertEqual(s3.size, 20)
            # Search should still work after reload
            results = s3.search(self.vecs[0], top_k=3)
            self.assertEqual(results[0].chunk_id, self.ids[0])


class TestQueryProcessor(unittest.TestCase):

    def test_preprocess_query(self):
        from core.query_processor import preprocess_query
        self.assertEqual(preprocess_query("  what is DNA?  "), "what is DNA?")
        # Filler removal
        result = preprocess_query("Can you please tell me what is DNA?")
        self.assertNotIn("can you", result.lower())
        self.assertNotIn("please", result.lower())

    def test_expand_query(self):
        from core.query_processor import expand_query
        variants = expand_query("Who invented the telephone?")
        self.assertGreater(len(variants), 1)
        self.assertIn("Who invented the telephone?", variants)

    def test_conversation_context(self):
        from core.query_processor import ConversationContext
        ctx = ConversationContext(window=2)
        ctx.add_turn("assistant", "Tom Brady plays for the Patriots.")
        ctx.add_turn("user", "How many rings does he have?")
        contextual = ctx.get_contextual_query("What is his salary?")
        self.assertIn("Patriots", contextual)

    def test_bm25_scorer(self):
        from core.query_processor import BM25Scorer
        corpus = [
            "Alexander Graham Bell invented the telephone.",
            "The telephone was a revolutionary invention.",
            "Paris is the capital of France.",
        ]
        bm25 = BM25Scorer()
        bm25.fit(corpus)
        score_relevant     = bm25.score("telephone inventor", corpus[0])
        score_irrelevant   = bm25.score("telephone inventor", corpus[2])
        self.assertGreater(score_relevant, score_irrelevant)


class TestResponseGenerator(unittest.TestCase):

    def test_validate_response(self):
        from core.response_generator import validate_response
        contexts = [{"text": "Bell invented the telephone in 1876.",
                     "metadata": {}}]

        # Good response
        v = validate_response("Bell invented the telephone.", "who invented it", contexts)
        self.assertGreater(v["confidence"], 0.5)

        # Too short
        v2 = validate_response("Bell", "who invented it", contexts)
        self.assertLess(v2["confidence"], 0.9)

    def test_fallback_extraction(self):
        from core.response_generator import ResponseGenerator
        gen = ResponseGenerator()
        contexts = [{"text": "Q: who invented the telephone\nA: Alexander Graham Bell",
                     "score": 0.9,
                     "chunk_id": "c1",
                     "metadata": {"chunk_type": "short_answer"}}]
        direct = gen._extract_direct_answer("who invented telephone", contexts)
        self.assertIn("Alexander Graham Bell", direct)

    def test_prompt_building(self):
        from core.response_generator import build_rag_prompt
        contexts = [{"text": "Bell invented the telephone.",
                     "score": 0.9,
                     "metadata": {"domain": "history", "difficulty": "easy"}}]
        msgs = build_rag_prompt("Who invented the telephone?", contexts)
        self.assertIsInstance(msgs, list)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertTrue(any(m["role"] == "user" for m in msgs))


class TestEvaluationMetrics(unittest.TestCase):

    def test_exact_match(self):
        from evaluation.metrics import exact_match
        self.assertEqual(exact_match("Paris", "Paris"), 1.0)
        self.assertEqual(exact_match("paris", "Paris"), 1.0)   # normalised
        self.assertEqual(exact_match("Lyon", "Paris"), 0.0)

    def test_token_f1(self):
        from evaluation.metrics import token_f1
        self.assertAlmostEqual(token_f1("Alexander Graham Bell", 
                                         "Alexander Graham Bell"), 1.0)
        f1 = token_f1("Alexander Bell", "Alexander Graham Bell")
        self.assertGreater(f1, 0.5)
        self.assertLess(f1, 1.0)

    def test_mrr(self):
        from evaluation.metrics import mean_reciprocal_rank
        retrieved = [["d3", "d1", "d2"], ["d2", "d3", "d1"]]
        relevant  = [{"d1"}, {"d2"}]
        mrr = mean_reciprocal_rank(retrieved, relevant)
        # Query 1: first relevant at rank 2 → 0.5
        # Query 2: first relevant at rank 1 → 1.0
        self.assertAlmostEqual(mrr, 0.75, places=3)

    def test_ndcg(self):
        from evaluation.metrics import ndcg_at_k
        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant  = {"d1", "d3"}
        ndcg = ndcg_at_k(retrieved, relevant, k=5)
        self.assertGreater(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

    def test_precision_recall(self):
        from evaluation.metrics import precision_at_k, recall_at_k
        retrieved = ["d1", "d2", "d3", "d4", "d5"]
        relevant  = {"d1", "d3", "d6"}
        self.assertAlmostEqual(precision_at_k(retrieved, relevant, 3), 2/3)
        self.assertAlmostEqual(recall_at_k(retrieved, relevant, 5), 2/3)


class TestIntegration(unittest.TestCase):
    """End-to-end integration test with sample data."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w")
        cls.tmp.close()
        make_sample_csv(cls.tmp.name, 15)

        from core.data_processor import NaturalQuestionsProcessor
        processor = NaturalQuestionsProcessor(cls.tmp.name)
        cls.chunks = processor.process(max_rows=15)

        from api.app import RAGPipeline
        cls.pipeline = RAGPipeline()
        cls.pipeline.build(cls.chunks)

    def test_simple_query(self):
        result = self.pipeline.query(
            "Who invented the telephone?",
            top_k=3, use_expansion=True, use_rerank=True
        )
        self.assertIsInstance(result["answer"], str)
        self.assertGreater(len(result["answer"]), 0)
        self.assertIn("retrieved", result)

    def test_query_returns_relevant_chunks(self):
        result = self.pipeline.query("capital of France", top_k=3)
        # Paris should appear somewhere in the retrieved texts
        all_text = " ".join(r["text"] for r in result["retrieved"])
        self.assertIn("Paris", all_text)

    def test_conversation_context(self):
        session = "test-session-123"
        self.pipeline.query(
            "What is photosynthesis?",
            use_conversation=True, session_id=session
        )
        result2 = self.pipeline.query(
            "What does it produce?",
            use_conversation=True, session_id=session
        )
        self.assertIsInstance(result2["answer"], str)

    def test_health_endpoint(self):
        health = self.pipeline.health()
        self.assertIn("status", health)
        self.assertEqual(health["status"], "ready")
        self.assertGreater(health["index_size"], 0)

    def test_score_threshold_filters(self):
        result_high = self.pipeline.query(
            "Who invented the telephone?",
            score_threshold=0.99   # very high – should get few/no results
        )
        result_low  = self.pipeline.query(
            "Who invented the telephone?",
            score_threshold=0.0    # no filter
        )
        self.assertLessEqual(
            len(result_high["retrieved"]),
            len(result_low["retrieved"])
        )


# ──────────────────────────────────────────────────────────────────────────────
#  Runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # quiet during tests
    unittest.main(verbosity=2)
