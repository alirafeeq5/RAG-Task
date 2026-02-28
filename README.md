# Multilingual RAG System

## Project Structure
```
multilingual_rag_system/
├── config.py                      # All settings & hyperparameters
├── main.py                        # Entry point – full demo
├── core/
│   ├── data_processor.py          # Phase 1 – Dataset processing, chunking, enrichment
│   ├── embeddings.py              # Phase 1 – Multilingual embeddings (ST + TF-IDF fallback)
│   ├── vector_store.py            # Phase 1 – FAISS + NumPy vector store + DocumentIndex
│   ├── query_processor.py         # Phase 2 – Query expansion, re-ranking, conversation
│   └── response_generator.py      # Phase 2 – Groq LLM integration, prompts, validation
├── api/
│   └── app.py                     # FastAPI REST API + RAGPipeline orchestrator
├── database/
│   └── models.py                  # SQLAlchemy models (documents, chunks, query/response logs)
├── utils/
│   └── performance.py             # Phase 2 – Monitoring, metrics, batch processor
├── evaluation/
│   └── metrics.py                 # MRR, NDCG, P@K, R@K, EM, Token-F1, BERTScore approx
└── tests/
    └── test_rag.py                # 34 unit + integration tests
```

## Requirements
```
# Production
sentence-transformers>=2.2.0   # multilingual embeddings
faiss-cpu>=1.7.0               # vector search
fastapi>=0.100.0               # REST API
uvicorn>=0.23.0                # ASGI server
sqlalchemy>=2.0.0              # ORM
pydantic>=2.0.0                # data validation
pandas>=1.5.0                  # data processing
numpy>=1.24.0                  # numerics
scikit-learn>=1.2.0            # TF-IDF, metrics
python-dotenv>=1.0.0           # env vars
gradio>=4.0.0                  # web UI & shareable link

# All available with offline fallbacks (sklearn TF-IDF + NumPy)
```

## Quick Start
```bash
pip install sentence-transformers faiss-cpu fastapi uvicorn sqlalchemy pydantic pandas numpy scikit-learn

# Set your Groq API key
export GROQ_API_KEY=your_key_here

# Run the full demo
python main.py --csv Book1.csv --rows 500 --queries 10

# Start the API server
python -c "
from main import run_demo
pipeline = run_demo('Book1.csv', max_rows=500)
from api.app import create_app
import uvicorn
app = create_app(pipeline)
uvicorn.run(app, host='0.0.0.0', port=8000)
"

# Run tests
python tests/test_rag.py
```

## Share via Link (Gradio Web UI)

The easiest way to share this system with others is the built-in **Gradio web interface**.
It generates a public `https://xxxxxx.gradio.live` link that you can send to anyone—no server setup required on their end.

```bash
# Install Gradio (one-time)
pip install gradio

# Launch the UI – a public shareable link is printed automatically
python gradio_app.py --csv data/Book1.csv --rows 200

# Local-only (no public link)
python gradio_app.py --csv data/Book1.csv --rows 200 --no-share

# Larger dataset / different port
python gradio_app.py --csv data/Book1.csv --rows 1000 --port 7861
```

After launch you will see something like:

```
🔗  A public link will be printed below — share it with anyone!

Running on public URL: https://a1b2c3d4e5f6.gradio.live
```

Copy that URL and share it. The link stays active for **72 hours**.

### Gradio UI Features
| Feature | Description |
|---------|-------------|
| Chat interface | Multi-turn conversation with history |
| Query expansion | Automatically expands your question |
| Re-ranking | BM25 + dense hybrid re-ranking |
| Multi-turn toggle | Enable/disable conversation memory |
| Confidence & latency | Shown below every answer |

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Single-turn RAG query |
| `POST` | `/conversation/query` | Multi-turn conversational RAG |
| `DELETE` | `/conversation/{id}` | Reset conversation context |
| `GET` | `/health` | System health + metrics |
| `GET` | `/stats` | Performance stats |
| `GET` | `/docs` | Swagger UI |

## Architecture

### Phase 1: Data Processing
- **Dataset Loading**: Handles the Natural Questions CSV with 307K rows
- **Text Cleaning**: HTML stripping, whitespace normalisation, punctuation fixing
- **Metadata Enrichment**: Auto-classifies question type (what/who/when/where/how/why), domain (9 categories), difficulty (easy/medium/hard), and language
- **Hybrid Chunking**: Sentence-boundary-aware sliding window with configurable overlap
- **Short/Long Answer Routing**: Short answers become high-confidence exact-match chunks; long answers are windowed

### Phase 1: Vector Store
- **Embeddings**: `paraphrase-multilingual-mpnet-base-v2` (50+ languages, 768-dim) with TF-IDF/LSA fallback
- **FAISS Index**: IVFFlat (configurable: Flat, HNSW) with cosine similarity
- **NumPy Fallback**: Pure NumPy brute-force for offline environments
- **Batch Indexing**: Configurable batch size with progress reporting

### Phase 2: Query Processing
- **Normalisation**: Filler phrase removal, whitespace collapse, CAPS detection
- **Expansion**: Synonym injection, keyword extraction, declarative form conversion
- **Multi-turn Context**: Sliding window of N prior turns injected into query
- **Hybrid Re-ranking**: BM25 (0.3 weight) + dense cosine (0.7 weight)
- **MMR Diversity**: Maximal Marginal Relevance to avoid redundant results

### Phase 2: Response Generation
- **Groq Integration**: OpenAI-compatible API with `llama3-8b-8192`
- **Prompt Engineering**: System prompt + context passages + conversation history
- **Quality Validation**: Hallucination detection, uncertainty detection, context overlap check
- **Fallback Cascade**: LLM → direct short-answer extraction → generic fallback message
- **Response Cache**: LRU in-memory + optional disk cache with TTL

### Phase 2: Performance
- **PerformanceMonitor**: Context-manager-based operation timing with p50/p95/p99
- **BatchProcessor**: Configurable batch embedding with error resilience
- **MetricsCollector**: Sliding window throughput + latency percentiles
- **Database Logging**: All queries and responses logged with SQLAlchemy (MySQL/SQLite)
