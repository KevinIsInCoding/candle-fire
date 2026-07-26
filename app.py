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


def _load_collection():
    try:
        return load_collection(CHROMA_DIR, CHROMA_COLLECTION)
    except Exception:
        _logger.warning("ChromaDB collection not found — running in demo mode (no data)")
        return None

_collection = _load_collection()
_graph = _load_graph()
_trials = _load_trials()
_client = anthropic.Anthropic()

_n_chunks = _collection.count() if _collection else 0
_n_trials = len(_trials)
_kg_nodes = _graph.number_of_nodes() if _graph else 0

# Experimental therapy landscape (offline-built artifact; loaded once)
import landscape as landscape_mod

_landscape = landscape_mod.load_landscape()
_landscape_classes = landscape_mod.class_names(_landscape)

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

    if _collection is None:
        history = history + [{"role": "user", "content": message}]
        history = history + [{"role": "assistant", "content": "⚠️ The knowledge base has not been loaded yet. The pipeline data (ChromaDB index, knowledge graph, papers) needs to be uploaded to this Space. Please contact the Space administrator."}]
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


def _landscape_select(class_name: str, phases):
    """When the class (or phase filter) changes: repopulate therapies and show the first one's detail."""
    labels = landscape_mod.therapy_labels(_landscape, class_name, phases)
    first = labels[0] if labels else None
    return (
        gr.update(choices=labels, value=first),
        landscape_mod.therapy_detail_md(_landscape, class_name, first or ""),
        landscape_mod.trials_table_html(_landscape, class_name, first or ""),
    )


def _phase_change(phases, class_name: str):
    """Phase filter: rebuild the sunburst and repopulate the current class's therapies."""
    therapy_update, detail, trials = _landscape_select(class_name, phases)
    return (landscape_mod.build_sunburst(_landscape, phases), therapy_update, detail, trials)


def _therapy_select(class_name: str, therapy_label: str):
    return (
        landscape_mod.therapy_detail_md(_landscape, class_name, therapy_label),
        landscape_mod.trials_table_html(_landscape, class_name, therapy_label),
    )


with gr.Blocks(title="Candle-Fire — ALS Research Intelligence") as demo:

    with gr.Tabs():

        with gr.Tab("💬 Ask"):
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
                    avatar_images=(None, "assets/flame.svg"),
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

        with gr.Tab("🧭 Therapy Landscape"):
            with gr.Column(elem_classes="container"):
                gr.Markdown(
                    "### 🧭 Experimental ALS Therapy Landscape\n"
                    "Explore experimental therapies by **mechanism class → therapy → clinical trials** "
                    "(recruiting & closed). Click a wedge to zoom; use the selectors for trial details."
                )
                if _landscape is None:
                    gr.Markdown(
                        "*Landscape not built yet — run `uv run python scripts/build_landscape.py`.*"
                    )
                else:
                    _init_class = _landscape_classes[0]
                    _init_labels = landscape_mod.therapy_labels(_landscape, _init_class)
                    _init_label = _init_labels[0] if _init_labels else None

                    sunburst = gr.Plot(landscape_mod.build_sunburst(_landscape), show_label=False)

                    phase_cb = gr.CheckboxGroup(
                        choices=landscape_mod.PHASE_OPTIONS, value=landscape_mod.PHASE_OPTIONS,
                        label="Filter by trial phase",
                        info="Show therapies with a trial in the selected phase(s). All selected = the whole picture.",
                    )

                    with gr.Row():
                        class_dd = gr.Dropdown(
                            choices=_landscape_classes, value=_init_class,
                            label="Mechanism class", scale=1,
                        )
                        therapy_dd = gr.Dropdown(
                            choices=_init_labels, value=_init_label,
                            label="Therapy", scale=1,
                        )

                    detail_md = gr.Markdown(
                        landscape_mod.therapy_detail_md(_landscape, _init_class, _init_label or "")
                    )
                    trials_html = gr.HTML(
                        landscape_mod.trials_table_html(_landscape, _init_class, _init_label or "")
                    )

                    class_dd.change(
                        _landscape_select, inputs=[class_dd, phase_cb],
                        outputs=[therapy_dd, detail_md, trials_html],
                    )
                    therapy_dd.change(
                        _therapy_select, inputs=[class_dd, therapy_dd],
                        outputs=[detail_md, trials_html],
                    )
                    phase_cb.change(
                        _phase_change, inputs=[phase_cb, class_dd],
                        outputs=[sunburst, therapy_dd, detail_md, trials_html],
                    )

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
        ssr_mode=False,  # HF experimental Node SSR 503s on this Space; serve classic app from :7860
        css=_CSS,
        theme=gr.themes.Soft(),
    )
