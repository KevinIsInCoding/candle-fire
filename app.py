"""Gradio web UI for candle-fire — physician-facing ALS research intelligence."""
from __future__ import annotations

import json
from pathlib import Path

import anthropic
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from agents.research_agent import stream_research_agent
from config import CHROMA_COLLECTION, CHROMA_DIR, GRAPH_PICKLE_PATH, HF_DATASET_REPO, TRIALS_PATH
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


_HF_DATASET = HF_DATASET_REPO


def _ensure_data() -> None:
    """Download chroma index + graph from the HF dataset repo if not already present.

    No-op locally (data is on disk); on the HF Space (empty storage) it fetches the
    runtime data. Having it here lets a single branch serve both dev and the Space.
    """
    need_chroma = not (CHROMA_DIR / "chroma.sqlite3").exists()
    need_graph = not GRAPH_PICKLE_PATH.exists()
    need_trials = not TRIALS_PATH.exists()
    if not need_chroma and not need_graph and not need_trials:
        return
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        if need_chroma:
            _logger.info("Downloading chroma index from HF dataset...")
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=_HF_DATASET, repo_type="dataset",
                local_dir=str(CHROMA_DIR), allow_patterns=["chroma/**"],
            )
            nested = CHROMA_DIR / "chroma"
            if nested.exists() and not (CHROMA_DIR / "chroma.sqlite3").exists():
                import shutil
                for item in nested.iterdir():
                    shutil.move(str(item), str(CHROMA_DIR / item.name))
                nested.rmdir()
            _logger.info("Chroma download complete")
        if need_graph:
            _logger.info("Downloading graph from HF dataset...")
            GRAPH_PICKLE_PATH.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=_HF_DATASET, repo_type="dataset",
                filename="graph/als_graph.pkl", local_dir=str(GRAPH_PICKLE_PATH.parent.parent),
            )
            _logger.info("Graph download complete")
        if need_trials:
            _logger.info("Downloading trials from HF dataset...")
            TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=_HF_DATASET, repo_type="dataset",
                filename="trials/trials.jsonl", local_dir=str(TRIALS_PATH.parent.parent),
            )
            _logger.info("Trials download complete")
    except Exception as e:
        _logger.warning(f"Failed to download data from HF dataset: {e}")


_ensure_data()

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
_mech_options = landscape_mod.mechanism_filter_options(_landscape)

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


def _refresh(status: str, phases: list, mech: str):
    """Any filter changed: redraw the wheel (rings = selected phases, narrowed by mechanism)."""
    labels = landscape_mod.compound_labels(_landscape, mech, status, phases)
    first = labels[0] if labels else None
    return (
        landscape_mod.build_pipeline_svg(_landscape, status, phases, mech),
        gr.update(choices=labels, value=first),
        landscape_mod.compound_detail_md(_landscape, first or ""),
        landscape_mod.compound_trials_html(_landscape, first or ""),
    )


def _compound_change(label: str):
    return (
        landscape_mod.compound_detail_md(_landscape, label),
        landscape_mod.compound_trials_html(_landscape, label),
    )


# ── Clinical Trials tab (facility / geography search) ─────────────────────────

import trials_query

_TRIAL_ENRICH_CAP = 25
_US_STATES = ["All"] + sorted(set(trials_query._STATE_ABBREV.values()))


def _search_trials(facility: str, state: str, city: str, status: str) -> str:
    facility = (facility or "").strip() or None
    city = (city or "").strip() or None
    state = None if (not state or state == "All") else state

    if not any([facility, city, state]):
        return '<div style="color:#888;padding:12px 0;">Enter a facility, state, or city to search.</div>'

    matches = trials_query.search_trials_by_location(
        _trials, facility=facility, city=city, state=state, status=status
    )
    enriched = [
        trials_query.enrich_trial(t, _collection, _graph, _trials)
        for t in matches[:_TRIAL_ENRICH_CAP]
    ]
    return trials_query.render_trials_html(enriched, len(matches))


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
                    "### 🧭 ALS Therapeutic Pipeline by Clinical Trial Phase\n"
                    "Mechanism groups are **sectors**; the three trial phases are **concentric "
                    "rings** (inner = Phase 1, outer = Phase 3). Each **dot is a compound** — "
                    "hover to see its name; **grey dots** have no recruiting/active trial. Use the "
                    "filters to narrow the wheel, and pick a **mechanism** to see its compounds' "
                    "pipeline stage, evidence confidence, and trials."
                )
                if _landscape is None:
                    gr.Markdown(
                        "*Landscape not built yet — run `uv run python scripts/build_landscape.py`.*"
                    )
                else:
                    _init_mech = landscape_mod.ALL_MECHANISMS
                    _init_labels = landscape_mod.compound_labels(_landscape, _init_mech)
                    _init_label = _init_labels[0] if _init_labels else None

                    with gr.Row():
                        status_dd = gr.Dropdown(
                            choices=landscape_mod.STATUS_FILTER_OPTIONS, value="All trials",
                            label="Recruitment status", scale=1,
                            info="Filter the wheel to compounds with a recruiting trial (or without one).",
                        )
                        phase_cb = gr.CheckboxGroup(
                            choices=landscape_mod.PHASE_RINGS, value=landscape_mod.PHASE_RINGS,
                            label="Trial phase", scale=1,
                            info="Each checked phase is drawn as a ring (inner → outer).",
                        )
                        mech_dd = gr.Dropdown(
                            choices=_mech_options, value=_init_mech,
                            label="Mechanism", scale=1,
                            info="Narrow the wheel to a single mechanism.",
                        )

                    wheel_html = gr.HTML(
                        landscape_mod.build_pipeline_svg(
                            _landscape, "All trials", landscape_mod.PHASE_RINGS, _init_mech)
                    )

                    compound_dd = gr.Dropdown(
                        choices=_init_labels, value=_init_label,
                        label="Compound — pipeline stage, evidence confidence & trials below",
                    )

                    detail_md = gr.Markdown(
                        landscape_mod.compound_detail_md(_landscape, _init_label or "")
                    )
                    trials_html = gr.HTML(
                        landscape_mod.compound_trials_html(_landscape, _init_label or "")
                    )

                    for _f in (status_dd, phase_cb, mech_dd):
                        _f.change(
                            _refresh, inputs=[status_dd, phase_cb, mech_dd],
                            outputs=[wheel_html, compound_dd, detail_md, trials_html],
                        )
                    compound_dd.change(
                        _compound_change, inputs=[compound_dd],
                        outputs=[detail_md, trials_html],
                    )

                gr.HTML(_DISCLAIMER_MD)

        with gr.Tab("🏥 Clinical Trials"):
            with gr.Column(elem_classes="container"):
                gr.Markdown(
                    "### 🏥 Find ALS Trials by Facility or Geography\n"
                    "Search the trial database by **facility** (e.g. *Mass General Hospital*) or "
                    "**location** (state / city). Each result is enriched with recruiting status, an "
                    "evidence-strength tier, key supporting papers, and related trials for the same compound."
                )
                with gr.Row():
                    facility_tb = gr.Textbox(
                        label="Facility / institution", scale=2,
                        placeholder="e.g. Mass General Hospital",
                    )
                    state_dd = gr.Dropdown(
                        choices=_US_STATES, value="All", label="State", scale=1,
                    )
                    city_tb = gr.Textbox(label="City", scale=1, placeholder="e.g. Boston")
                    trial_status_dd = gr.Dropdown(
                        choices=["All", "Recruiting", "Not recruiting"],
                        value="All", label="Recruitment status", scale=1,
                    )
                search_btn = gr.Button("Search trials", variant="primary")
                trial_results = gr.HTML(
                    '<div style="color:#888;padding:12px 0;">Enter a facility, state, or city to search.</div>'
                )

                _trial_search_inputs = [facility_tb, state_dd, city_tb, trial_status_dd]
                search_btn.click(_search_trials, inputs=_trial_search_inputs, outputs=[trial_results])
                facility_tb.submit(_search_trials, inputs=_trial_search_inputs, outputs=[trial_results])
                city_tb.submit(_search_trials, inputs=_trial_search_inputs, outputs=[trial_results])

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
