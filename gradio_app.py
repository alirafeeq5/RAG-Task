"""
Gradio Web UI  –  Multilingual RAG System
==========================================
Launches a browser-based chat interface for the RAG pipeline.
Run:
    python gradio_app.py --csv data/Book1.csv --rows 200
    python gradio_app.py --csv data/Book1.csv --rows 200 --share   # public URL

The --share flag (on by default) prints a public link like
    https://xxxxxx.gradio.live
that you can send to anyone.
"""

import argparse
import logging
import sys
import os
import uuid

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import CONFIG, DATA_DIR

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("gradio_app")


# ── pipeline (lazy-loaded once at startup) ────────────────────────────────────
_pipeline = None


def get_pipeline(csv_path: str, max_rows: int):
    """Build and cache the RAG pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from core.data_processor import NaturalQuestionsProcessor
    from api.app import RAGPipeline

    processor = NaturalQuestionsProcessor(csv_path)
    processor.load(max_rows=max_rows)
    processor.enrich()
    processor.chunk()

    from database.models import DatabaseManager
    db = DatabaseManager()
    db.save_documents(processor.enriched)
    db.save_chunks(processor.chunks)

    _pipeline = RAGPipeline()
    _pipeline.build(processor.chunks, batch_size=128)
    return _pipeline


# ── chat logic ────────────────────────────────────────────────────────────────

def respond(message: str, history: list, session_id: str,
            top_k: int, use_expansion: bool, use_rerank: bool,
            use_conversation: bool):
    """Called by Gradio on every chat message."""
    if not message.strip():
        return history, session_id

    pipeline = _pipeline
    if pipeline is None:
        history = history + [(message, "⚠️ Pipeline not initialised yet. Please wait and retry.")]
        return history, session_id

    if not session_id:
        session_id = str(uuid.uuid4())

    result = pipeline.query(
        question         = message,
        top_k            = top_k,
        use_expansion    = use_expansion,
        use_rerank       = use_rerank,
        use_conversation = use_conversation,
        session_id       = session_id if use_conversation else None,
    )

    answer = result["answer"]
    meta = (
        f"\n\n---\n*Confidence: {result['confidence']:.2f} | "
        f"Sources: {len(result['retrieved'])} | "
        f"Latency: {result['latency_ms']} ms*"
    )
    history = history + [(message, answer + meta)]
    return history, session_id


def clear_session(session_id: str):
    """Reset the conversation session."""
    if _pipeline and session_id:
        _pipeline.reset_session(session_id)
    return [], str(uuid.uuid4())


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui():
    try:
        import gradio as gr
    except ImportError:
        raise RuntimeError("Gradio is not installed. Run: pip install gradio")

    with gr.Blocks(title="Multilingual RAG System", theme=gr.themes.Soft()) as demo:

        gr.Markdown(
            """
# 🔍 Multilingual RAG System
Ask questions against the knowledge base. Supports multi-turn conversations.
> Share this app with anyone using the **public link** printed in your terminal.
            """
        )

        session_state = gr.State(str(uuid.uuid4()))

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=450)
                msg_box = gr.Textbox(
                    placeholder="Type your question here and press Enter …",
                    label="Your question",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn  = gr.Button("Clear / New session")

            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### ⚙️ Settings")
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1, label="Top-K results"
                )
                use_expansion_cb  = gr.Checkbox(value=True,  label="Query expansion")
                use_rerank_cb     = gr.Checkbox(value=True,  label="Re-ranking")
                use_conv_cb       = gr.Checkbox(value=False, label="Multi-turn conversation")

        # ── wire events ───────────────────────────────────────────────────────
        def on_submit(message, history, session_id, top_k,
                      use_expansion, use_rerank, use_conv):
            new_history, new_session = respond(
                message, history, session_id,
                top_k, use_expansion, use_rerank, use_conv,
            )
            return new_history, new_session, ""   # clear the textbox

        submit_btn.click(
            fn      = on_submit,
            inputs  = [msg_box, chatbot, session_state,
                       top_k_slider, use_expansion_cb, use_rerank_cb, use_conv_cb],
            outputs = [chatbot, session_state, msg_box],
        )
        msg_box.submit(
            fn      = on_submit,
            inputs  = [msg_box, chatbot, session_state,
                       top_k_slider, use_expansion_cb, use_rerank_cb, use_conv_cb],
            outputs = [chatbot, session_state, msg_box],
        )
        clear_btn.click(
            fn      = lambda sid: clear_session(sid),
            inputs  = [session_state],
            outputs = [chatbot, session_state],
        )

    return demo


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Gradio Web UI")
    parser.add_argument("--csv",    default=str(DATA_DIR / "Book1.csv"),
                        help="Path to the Natural Questions CSV")
    parser.add_argument("--rows",   type=int, default=200,
                        help="Max rows to load from CSV (default: 200)")
    parser.add_argument("--no-share", dest="share", action="store_false",
                        help="Disable the public Gradio link (local only)")
    parser.set_defaults(share=True)
    parser.add_argument("--port",   type=int, default=7860,
                        help="Local port (default: 7860)")
    args = parser.parse_args()

    print(f"\n⏳  Loading dataset and building index ({args.rows} rows) …")
    get_pipeline(args.csv, args.rows)
    print("✅  Pipeline ready.\n")

    demo = build_ui()

    print("🚀  Launching Gradio …")
    if args.share:
        print("🔗  A public link will be printed below — share it with anyone!\n")

    demo.launch(
        server_name = "0.0.0.0",
        server_port = args.port,
        share       = args.share,
    )


if __name__ == "__main__":
    main()
