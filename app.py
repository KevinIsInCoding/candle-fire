"""Gradio web UI for candle-fire — physician-facing ALS research intelligence."""
from __future__ import annotations

import html
import json
import os
from concurrent.futures import ThreadPoolExecutor
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


# UI smoke mode (CANDLE_UI_SMOKE=1): render the Blocks WITHOUT the heavy startup loads
# (cross-encoder, ChromaDB, graph, Anthropic client) so UI/layout tests boot in seconds.
# The Clinical Trials tab only needs the trials list, which loads fast. Query features are
# inert in this mode — it exists purely to render and drive the interface.
_SMOKE = bool(os.getenv("CANDLE_UI_SMOKE"))

if not _SMOKE:
    _ensure_data()

_collection = None if _SMOKE else _load_collection()
_graph = None if _SMOKE else _load_graph()
_trials = _load_trials()
def _make_llm_client():
    """Runtime LLM client. LLM_PROVIDER=bedrock → Amazon Bedrock (HIPAA path,
    IAM auth via the instance role); otherwise the direct Anthropic API. Both
    expose the same messages.stream() surface used by the research agent."""
    provider = os.getenv("LLM_PROVIDER", "anthropic")
    if provider == "bedrock":
        # Legacy bedrock-runtime InvokeModel path (bedrock:InvokeModel*), driving
        # the us.anthropic.claude-sonnet-4-6 cross-region inference profile. The
        # newer Mantle endpoint needs a Bedrock "project" this account doesn't have.
        from anthropic import AnthropicBedrock
        region = os.getenv("AWS_REGION", "us-east-1")
        client = AnthropicBedrock(aws_region=region)
        _logger.info("LLM client: Amazon Bedrock in %s", region)
        return client
    return anthropic.Anthropic()


_client = None if _SMOKE else _make_llm_client()

_n_chunks = _collection.count() if _collection else 0
# Headline trial count = recruiting, interventional trials only (the actionable set), not the
# full corpus (which includes completed/terminated studies and expanded-access programs).
_RECRUITING = {"RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION", "AVAILABLE"}
_n_trials = sum(
    1 for t in _trials
    if t.get("study_type") == "INTERVENTIONAL" and t.get("status") in _RECRUITING
)
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
    history = history + [{"role": "assistant", "content": "*Analyzing your question...*"}]
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
/* Autocomplete gate: hide a combobox's attached option list until ≥3 chars (see
   _AUTOCOMPLETE_GATE_JS). The script toggles .ac-hide on the input's wrapper by length. */
#facility_combo.ac-hide ul, #city_combo.ac-hide ul { display: none !important; }
/* Smaller filter labels so long ones (e.g. "Recruitment status") stay on one line and the
   dropdown chevron doesn't overlap the text. */
.trial-filters label span { font-size: 0.78rem !important; white-space: nowrap; }
"""

_TITLE_MD = """# 🕯️ Candle-Fire
### ALS Research Intelligence for Physicians
Ask a free-text question about ALS biology, drug targets, or clinical trials.
Answers are synthesized from a curated ALS research corpus and enriched by a biomedical knowledge graph.
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

# Facility/city autocomplete vocabulary (built once from the trials' site data), ranked
# busiest-first and formatted as combobox (label, value) choices.
_LOC_INDEX = trials_query.build_location_index(_trials)
_LOC_CHOICES = trials_query.location_choices(_LOC_INDEX)


def _warm_trial_cache() -> None:
    """Prewarm the per-trial supporting-papers cache in the background so Clinical
    Trials searches don't pay the CPU-bound ChromaDB lookup in the request path.
    Runs after launch; searches that arrive before it finishes just fill the cache
    themselves."""
    if _collection is None:
        return
    import time as _time
    start = _time.time()
    for _rec in _trials:
        try:
            trials_query._find_supporting_papers(_rec, _collection)
        except Exception:
            pass
    _logger.info("Trial supporting-papers cache warmed: %d trials in %.0fs",
                 len(_trials), _time.time() - start)


if not _SMOKE and _collection is not None:
    import threading
    threading.Thread(target=_warm_trial_cache, daemon=True).start()

# Combobox behavior that Gradio can't express natively, wired in JS:
#   - an in-box placeholder hint (gr.Dropdown has no `placeholder` param), and
#   - hiding the attached option list until MIN_AUTOCOMPLETE_CHARS (a `.ac-hide` class the
#     CSS in _CSS acts on).
# Injected via gr.Blocks(head=...) as a real <script> — Gradio's js=/demo.load(js=) callbacks
# did not execute in this version, but a <head> script runs directly in the browser. The
# Clinical Trials tab renders LAZILY (inputs appear only when the tab is first opened), so a
# MutationObserver re-runs the wiring as the DOM changes and marks each input done, attaching
# whenever the tab renders. Degrades gracefully — if the dropdown DOM differs, the combobox
# still filters from the first character.
_AUTOCOMPLETE_GATE_HEAD = f"""
<script>
(function() {{
  const MIN = {trials_query.MIN_AUTOCOMPLETE_CHARS};
  const targets = {{
    'facility_combo': 'Type \\u22653 letters, e.g. Mass General',
    'city_combo': 'Type \\u22653 letters, busiest cities first',
  }};
  const gate = (id, hint) => {{
    const root = document.getElementById(id);
    if (!root) return;
    const input = root.querySelector('input');
    if (!input || input.dataset.acReady) return;   // not rendered yet, or already wired
    input.dataset.acReady = '1';
    if (hint) input.setAttribute('placeholder', hint);
    const apply = () => root.classList.toggle('ac-hide', input.value.trim().length < MIN);
    input.addEventListener('input', apply);
    input.addEventListener('focus', apply);
    apply();
  }};
  const applyAll = () => Object.entries(targets).forEach(([id, h]) => gate(id, h));
  const start = () => {{
    applyAll();
    new MutationObserver(applyAll).observe(document.body, {{childList: true, subtree: true}});
  }};
  if (document.body) start();
  else document.addEventListener('DOMContentLoaded', start);
}})();
</script>
"""


def _search_trials(facility: str, state: str, city: str, study_type: str, status: str) -> str:
    facility = (facility or "").strip() or None
    city = (city or "").strip() or None
    state = None if (not state or state == "All") else state

    if not any([facility, city, state]):
        return '<div style="color:#888;padding:12px 0;">Enter a facility, state, or city to search.</div>'

    matches = trials_query.search_trials_by_location(
        _trials, facility=facility, city=city, state=state,
        status=status, study_type=study_type,
    )

    # If the active filters hide everything, say whether broader filters would find trials —
    # e.g. a facility with only completed studies under the default Recruiting + Interventional.
    if not matches and (status != "All" or study_type != "All"):
        broad = trials_query.search_trials_by_location(
            _trials, facility=facility, city=city, state=state, status="All", study_type="All",
        )
        if broad:
            where = ", ".join(p for p in (facility, city, state) if p)
            return (
                '<div style="background:#fff6e5;border:1px solid #ffe0a3;border-radius:8px;'
                'padding:10px 12px;margin:6px 0;color:#7a5b00;font-size:0.9rem;">'
                f'No <b>{html.escape((study_type or "").lower())}</b> trials that are '
                f'<b>{html.escape((status or "").lower())}</b> at {html.escape(where)}. '
                f'{len(broad)} trial(s) exist there under broader filters — set '
                '<b>Study type</b> and <b>Recruitment status</b> to <b>All</b> to see them.'
                '</div>'
            )

    # Enrich concurrently: each trial does an independent ChromaDB lookup, so a small
    # thread pool cuts the wall time roughly in half on the 2-vCPU box. Build the
    # mechanism index once (it's cached anyway) and pass it in. ex.map preserves order.
    to_enrich = matches[:_TRIAL_ENRICH_CAP]
    mech_index = trials_query._mechanism_index()
    with ThreadPoolExecutor(max_workers=4) as ex:
        enriched = list(ex.map(
            lambda t: trials_query.enrich_trial(t, _collection, _graph, _trials, mechanism_index=mech_index),
            to_enrich,
        ))
    return trials_query.render_trials_html(enriched, len(matches))


with gr.Blocks(title="Candle-Fire — ALS Research Intelligence", head=_AUTOCOMPLETE_GATE_HEAD) as demo:

    with gr.Tabs():

        with gr.Tab("💬 Ask"):
            with gr.Column(elem_classes="container"):

                gr.Markdown(_TITLE_MD)

                gr.HTML(
                    f'<div class="status-bar">'
                    f'{_n_chunks} paper chunks &nbsp;·&nbsp; '
                    f'{_n_trials} recruiting interventional trials &nbsp;·&nbsp; '
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
                with gr.Row(elem_classes="trial-filters"):
                    facility_tb = gr.Dropdown(
                        choices=_LOC_CHOICES["facilities"], value=None,
                        label="Facility / institution", scale=2,
                        filterable=True, allow_custom_value=True, elem_id="facility_combo",
                    )
                    state_dd = gr.Dropdown(
                        choices=_US_STATES, value="All", label="State", scale=1,
                    )
                    city_tb = gr.Dropdown(
                        choices=_LOC_CHOICES["cities"], value=None,
                        label="City", scale=1,
                        filterable=True, allow_custom_value=True, elem_id="city_combo",
                    )
                # Filters on their own row so the labels/values have full width — no wrapping,
                # no value running under the chevron.
                with gr.Row(elem_classes="trial-filters"):
                    study_type_dd = gr.Dropdown(
                        choices=["Interventional", "Expanded Access", "All"],
                        value="Interventional", label="Study type", scale=1,
                    )
                    trial_status_dd = gr.Dropdown(
                        choices=["All", "Recruiting", "Not recruiting"],
                        value="Recruiting", label="Recruitment status", scale=1,
                    )
                search_btn = gr.Button("Search trials", variant="primary")
                trial_results = gr.HTML(
                    '<div style="color:#888;padding:12px 0;">Enter a facility, state, or city to search.</div>'
                )

                # Facility/city are typeable comboboxes (filterable Dropdowns) — the physician
                # types and picks from the attached, busiest-first list. No per-keystroke server
                # event needed; the search reads the selected/typed value directly.
                _trial_search_inputs = [facility_tb, state_dd, city_tb, study_type_dd, trial_status_dd]
                search_btn.click(_search_trials, inputs=_trial_search_inputs, outputs=[trial_results])
                # Picking a facility/city from its list also runs the search immediately.
                facility_tb.select(_search_trials, inputs=_trial_search_inputs, outputs=[trial_results])
                city_tb.select(_search_trials, inputs=_trial_search_inputs, outputs=[trial_results])

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
