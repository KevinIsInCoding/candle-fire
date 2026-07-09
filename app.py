"""Gradio web UI for candle-fire — physician-facing ALS research intelligence."""
from __future__ import annotations

import json
from pathlib import Path

import anthropic
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from agents.research_agent import stream_research_agent
from config import CHROMA_COLLECTION, CHROMA_DIR, GRAPH_PICKLE_PATH, TRIALS_PATH
from logging_config import get_logger
from rag.indexer import load_collection

_logger = get_logger("app")

# ── Load resources once at startup ───────────────────────────────────────────

def _load_graph():
    try:
        from graph.serializer import load_graph
        G = load_graph(GRAPH_PICKLE_PATH)
        _logger.info(f"KG loaded: {G.number_of_nodes()} nodes")
        return G
    except FileNotFoundError:
        _logger.warning("KG not found — running RAG-only mode")
        return None


def _load_trials() -> list[dict]:
    if not TRIALS_PATH.exists():
        return []
    with open(TRIALS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_collection = load_collection(CHROMA_DIR, CHROMA_COLLECTION)
_graph = _load_graph()
_trials = _load_trials()
_client = anthropic.Anthropic()

_n_chunks = _collection.count()
_n_trials = len(_trials)
_kg_nodes = _graph.number_of_nodes() if _graph else 0

# ── Example questions ─────────────────────────────────────────────────────────

_EXAMPLES = [
    "What is the evidence for tofersen targeting SOD1 in ALS?",
    "What mechanisms link TDP-43 aggregation to motor neuron death?",
    "What compounds target glutamate excitotoxicity in ALS?",
    "What is the role of C9orf72 repeat expansion in neurodegeneration?",
    "How does riluzole work and what is the clinical evidence?",
    "What biomarkers track ALS disease progression?",
]

# ── Streaming respond function ────────────────────────────────────────────────

def respond(message: str, history: list[dict]):
    if not message.strip():
        yield history, gr.update(value="", interactive=True)
        return

    history = history + [{"role": "user", "content": message}]
    history = history + [{"role": "assistant", "content": ""}]
    yield history, gr.update(value="", interactive=False)

    response_text = ""

    for event_type, content in stream_research_agent(
        _client, message, _collection, _trials, graph=_graph
    ):
        if event_type == "status":
            if not response_text:
                history[-1]["content"] = f"*{content}*"
                yield history, gr.update()
        elif event_type == "token":
            response_text += content
            history[-1]["content"] = response_text
            yield history, gr.update()
        elif event_type == "done":
            history[-1]["content"] = response_text or content
            yield history, gr.update(interactive=True)
            return

    yield history, gr.update(interactive=True)


# ── UI ────────────────────────────────────────────────────────────────────────

_CSS = """
.container { max-width: 900px; margin: 0 auto; }
.disclaimer { font-size: 0.78rem; color: #888; text-align: center; margin-top: 6px; }
.status-bar { font-size: 0.82rem; color: #666; text-align: center; margin-bottom: 8px; }
footer { display: none !important; }
"""

_TITLE_MD = """# 🕯️ Candle-Fire
### ALS Research Intelligence for Physicians
Ask a free-text question about ALS biology, drug targets, or clinical trials.
Answers are synthesized from ~500 curated ALS papers and enriched by a biomedical knowledge graph.
"""

_DISCLAIMER_MD = """<div class="disclaimer">
⚕️ Research synthesis tool — not a substitute for clinical judgment.
Always verify claims with primary sources before applying to patient care.
</div>"""


with gr.Blocks(title="Candle-Fire — ALS Research Intelligence") as demo:

    with gr.Column(elem_classes="container"):

        gr.Markdown(_TITLE_MD)

        gr.HTML(
            f'<div class="status-bar">'
            f'{_n_chunks} paper chunks &nbsp;·&nbsp; '
            f'{_n_trials} clinical trials &nbsp;·&nbsp; '
            f'{_kg_nodes} knowledge graph nodes'
            f'</div>'
        )

        chatbot = gr.Chatbot(
            value=[],
            height=520,
            show_label=False,
            sanitize_html=False,
            avatar_images=(None, "https://api.dicebear.com/7.x/icons/svg?seed=candle&icon=flame"),
            placeholder="Ask a question about ALS research to get started.",
        )

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="e.g. What is the evidence for tofersen targeting SOD1?",
                show_label=False,
                scale=9,
                autofocus=True,
                lines=1,
            )
            send_btn = gr.Button("Ask", scale=1, variant="primary", min_width=80)

        gr.Markdown("**Example questions** — click to populate:")

        with gr.Row():
            with gr.Column(scale=1):
                for ex in _EXAMPLES[:3]:
                    btn = gr.Button(ex, size="sm", variant="secondary")
                    btn.click(fn=lambda t=ex: t, outputs=[msg_box])
            with gr.Column(scale=1):
                for ex in _EXAMPLES[3:]:
                    btn = gr.Button(ex, size="sm", variant="secondary")
                    btn.click(fn=lambda t=ex: t, outputs=[msg_box])

        gr.HTML(_DISCLAIMER_MD)

    submit_kwargs = dict(
        fn=respond,
        inputs=[msg_box, chatbot],
        outputs=[chatbot, msg_box],
    )
    msg_box.submit(**submit_kwargs)
    send_btn.click(**submit_kwargs)


if __name__ == "__main__":
    demo.launch(
        share=False,
        css=_CSS,
        theme=gr.themes.Soft(),
    )
