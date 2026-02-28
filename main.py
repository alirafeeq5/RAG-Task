"""
Main Entry Point - Multilingual RAG System Demo
================================================
Runs the complete pipeline end-to-end:
  1. Load & process Natural Questions CSV
  2. Enrich with metadata
  3. Chunk documents
  4. Persist to database
  5. Build embedding index
  6. Run sample queries
  7. Print evaluation report
  8. Display performance metrics

Usage:  python main.py [--csv path] [--rows N] [--queries N]
"""

import sys
import os
import argparse
import logging
import json
import time

# ── make sure package root is on path ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import CONFIG, DATA_DIR

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers= [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(CONFIG.logging.log_dir / CONFIG.logging.filename),
                            encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def print_banner():
    banner = """
+==================================================================+
|         MULTILINGUAL RAG SYSTEM  -  AI Platform Component        |
|                                                                  |
|  Phase 1: Dataset Processing + Embeddings + FAISS Indexing       |
|  Phase 2: Query Expansion + Re-ranking + LLM Generation +        |
|           Multi-turn Conversation + Caching + Performance       |
+==================================================================+
"""
    print(banner)


def run_demo(csv_path: str, max_rows: int = 200, n_queries: int = 5):

    print_banner()
    t_start = time.time()

    # ══════════════════════════════════════════════════════════════
    # PHASE 1 - Data Processing
    # ══════════════════════════════════════════════════════════════
    print("\n" + "-"*60)
    print("  PHASE 1 - Dataset Acquisition & Processing")
    print("-"*60)

    from core.data_processor import NaturalQuestionsProcessor
    processor = NaturalQuestionsProcessor(csv_path)

    print(f"\n[1/4] Loading dataset  ({max_rows} rows) ...")
    processor.load(max_rows=max_rows)

    print("[2/4] Enriching records (type / domain / difficulty / language) ...")
    processor.enrich()

    print("[3/4] Chunking documents (hybrid strategy) ...")
    processor.chunk()

    chunks   = processor.chunks
    enriched = processor.enriched

    print("[4/4] Dataset statistics:")
    processor.statistics()

    # ══════════════════════════════════════════════════════════════
    # Database persistence
    # ══════════════════════════════════════════════════════════════
    print("\n" + "-"*60)
    print("  Persisting to Database")
    print("-"*60)
    from database.models import DatabaseManager
    db = DatabaseManager()
    saved_docs   = db.save_documents(enriched)
    saved_chunks = db.save_chunks(chunks)
    print(f"  Saved {saved_docs:,} documents and {saved_chunks:,} chunks to DB.")
    print(f"  DB stats: {json.dumps(db.get_stats(), indent=4)}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1 - Embedding + Indexing
    # ══════════════════════════════════════════════════════════════
    print("\n" + "-"*60)
    print("  PHASE 1 - Embedding & Vector Store Indexing")
    print("-"*60)

    from api.app import RAGPipeline
    pipeline = RAGPipeline()
    print(f"\n  Embedding backend : {pipeline._engine.backend_name}")
    print(f"  Vector store      : {type(pipeline._store).__name__}")
    print(f"  Embedding dim     : {pipeline._engine.dim}")

    print(f"\n  Indexing {len(chunks):,} chunks ...")
    pipeline.build(chunks, batch_size=128)
    print(f"  OK Index ready - {pipeline._index.size:,} vectors")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2 - Query Processing & Generation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "-"*60)
    print("  PHASE 2 - Query Processing, Expansion & Response Generation")
    print("-"*60)

    # Sample queries drawn from the dataset
    sample_queries = [
        "Who invented the telephone?",
        "What is the capital of France?",
        "When was the first iPhone released?",
        "How does photosynthesis work?",
        "What is opt-in email marketing?",
        "Who is the mother in How I Met Your Mother?",
        "What type of fertilisation occurs in humans?",
        "Who has the most NFL wins?",
    ][:n_queries]

    results_log = []
    for i, q in enumerate(sample_queries, 1):
        print(f"\n  [{i}/{len(sample_queries)}] Query: \"{q}\"")
        result = pipeline.query(
            question        = q,
            top_k           = 5,
            use_expansion   = True,
            use_rerank      = True,
            use_conversation= False,
        )
        print(f"    Expanded to  : {result['expanded_queries']}")
        print(f"    Retrieved    : {len(result['retrieved'])} chunks")
        if result['retrieved']:
            top = result['retrieved'][0]
            print(f"    Top chunk    : score={top['score']:.3f} | "
                  f"domain={top['metadata'].get('domain','?')} | "
                  f"type={top['metadata'].get('chunk_type','?')}")
        print(f"    Answer       : {result['answer'][:200]}")
        print(f"    Confidence   : {result['confidence']} | "
              f"Valid={result['is_valid']} | "
              f"Fallback={result['used_fallback']}")
        print(f"    Latency      : {result['latency_ms']} ms")
        results_log.append(result)

    # ══════════════════════════════════════════════════════════════
    # Multi-turn conversation demo
    # ══════════════════════════════════════════════════════════════
    print("\n" + "-"*60)
    print("  PHASE 2 - Multi-Turn Conversation Demo")
    print("-"*60)
    session = "demo-session-001"
    conv_qs = [
        "What is opt-in email marketing?",
        "Give me an example of that.",
        "How is it different from spam?",
    ]
    for turn, cq in enumerate(conv_qs, 1):
        print(f"\n  Turn {turn}: \"{cq}\"")
        cr = pipeline.query(
            cq,
            use_conversation = True,
            session_id       = session,
        )
        print(f"    Answer  : {cr['answer'][:200]}")
        print(f"    Context : {cr.get('context_hint','')}")

    # ══════════════════════════════════════════════════════════════
    # Evaluation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "-"*60)
    print("  PHASE 2 - Evaluation Metrics")
    print("-"*60)

    from evaluation.metrics import RAGEvaluator, EvalSample, token_f1, exact_match

    # Build eval set from the dataset itself (question → short_answer)
    eval_samples = []
    for rec in enriched[:20]:
        if rec.has_short and rec.short_answer:
            eval_samples.append(EvalSample(
                query        = rec.question,
                ground_truth = rec.short_answer,
            ))
    eval_samples = eval_samples[:10]

    if eval_samples:
        print(f"\n  Evaluating on {len(eval_samples)} samples ...")
        evaluator = RAGEvaluator(pipeline)
        eval_results = evaluator.evaluate(eval_samples, k=5, verbose=True)
        report = RAGEvaluator.report(eval_results)
    else:
        print("  No short-answer samples available for evaluation.")

    # ══════════════════════════════════════════════════════════════
    # Performance Report
    # ══════════════════════════════════════════════════════════════
    print("\n" + "-"*60)
    print("  PHASE 2 - Performance & System Metrics")
    print("-"*60)
    health = pipeline.health()
    print(f"\n  System status     : {health['status']}")
    print(f"  Embedding backend : {health['backend']}")
    print(f"  Index size        : {health['index_size']:,} vectors")
    print(f"  Query cache       : {health['cache_stats']}")
    print(f"  LLM stats         : {health['llm_stats']}")
    if health['perf_metrics']:
        print(f"\n  Performance breakdown:")
        for op, metrics in health['perf_metrics'].items():
            if metrics:
                lat = metrics.get('latency_ms', {})
                print(f"    {op:<20} count={metrics.get('count',0):4d}  "
                      f"p50={lat.get('p50',0):.1f}ms  "
                      f"p95={lat.get('p95',0):.1f}ms  "
                      f"err={metrics.get('error_rate',0):.1%}")

    total_elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  OK Pipeline demo complete in {total_elapsed:.1f}s")
    print(f"{'='*60}\n")
    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual RAG System Demo")
    parser.add_argument("--csv",     default=str(DATA_DIR / "Book1.csv"),
                        help="Path to Natural Questions CSV")
    parser.add_argument("--rows",    type=int, default=200,
                        help="Max rows to process (default: 200)")
    parser.add_argument("--queries", type=int, default=6,
                        help="Number of demo queries (default: 6)")
    args = parser.parse_args()

    pipeline = run_demo(
        csv_path  = args.csv,
        max_rows  = args.rows,
        n_queries = args.queries,
    )
