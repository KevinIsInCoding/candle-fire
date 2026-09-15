# Candle-Fire — Architecture Guide

## What This Is

Candle-fire is a physician-facing ALS research intelligence tool. A physician asks a free-text question ("What's the evidence for tofersen targeting SOD1?") and gets a synthesized, cited answer grounded in ~10,000 curated ALS papers, enriched by a knowledge graph.

**Sibling project**: beacon (patient-facing clinical trial finder at `../beacon`). Follow the same conventions.

## Two-Layer Intelligence

1. **Knowledge Graph (KG)**: NetworkX DiGraph linking Gene → Protein → Compound → Pathway → Phenotype → ClinicalTrial. Used to expand query entities before retrieval (e.g., "tofersen" → SOD1 → oxidative stress → related compounds).

2. **RAG (Vector Search)**: ChromaDB collection of ~10,000 ALS paper abstracts/full-text. Citation-count-weighted re-ranking. Used to retrieve evidence passages for synthesis.

**Query pipeline**: KG expansion first, then RAG retrieval with expanded entity context, then Claude synthesis.

## Module Responsibilities

| File/Dir | Responsibility |
|---|---|
| `config.py` | All constants: model names, file paths, ALS seed entities, API endpoints |
| `models.py` | Dataclasses: `ALSPaper`, `ExtractedEntity`, `EntityRelationship`, `ResearchLandscape` |
| `prompts.py` | System prompts for extraction agent and synthesis agent |
| `tools.py` | Tool schema loader — reads JSON from `data/tools/`, exports typed tool params |
| `llm.py` | LLM provider abstraction (Anthropic/OpenAI switchable via `LLM_PROVIDER` env var) |
| `logging_config.py` | Structured JSON rotating log to `logs/candle_fire.log` |
| `ingestion/pubmed.py` | PubMed Entrez client: fetch abstracts + metadata by MeSH query or PMID list |
| `ingestion/pmc.py` | PMC XML full-text fetcher: structured section text for Open Access papers |
| `ingestion/clinicaltrials.py` | ClinicalTrials.gov v2 client for ALS trials (no geo/distance, unlike beacon) |
| `ingestion/semantic_scholar.py` | Citation count enrichment per PMID via Semantic Scholar API |
| `extraction/extractor.py` | Claude Sonnet NER: batch 10 papers/call, exponential backoff, resumable via `.progress.json` |
| `extraction/normalizer.py` | Entity name → canonical ID: HGNC alias table → PubChem → REST fallback |
| `graph/builder.py` | Build NetworkX DiGraph from `entities.jsonl`; upsert nodes+edges; citation-weighted confidence |
| `graph/query.py` | Graph traversal: `expand_query_entities()`, `find_trials_for_target()`, `get_entity_evidence()` |
| `graph/serializer.py` | Save/load graph: pickle (fast load at startup) + JSON (human-readable export) |
| `rag/indexer.py` | Build ChromaDB collection; section-aware chunking; `citation_count` in metadata |
| `rag/retriever.py` | `search()`, `search_by_entities()`, citation-weighted re-ranking |
| `agents/research_agent.py` | Multi-step synthesis agent (streaming): entity extraction → KG expansion → RAG → synthesis |
| `app.py` | Gradio UI (tabbed: Ask + Therapy Landscape): loads graph + ChromaDB + landscape once at startup |
| `landscape.py` | Therapy Landscape rendering: Plotly sunburst + detail-panel HTML from `landscape.json` |
| `main.py` | CLI interface (Rich console) |
| `scripts/` | Offline pipeline scripts: run once in order (ingest → extract → build_graph → build_index → build_landscape) |

## Offline Pipeline Run Order

Run these once to build the knowledge assets. Each is resumable.

```bash
# 1. Ingest papers from PubMed + PMC full text + Semantic Scholar citation counts
uv run python scripts/ingest_papers.py

# 2. Ingest ALS clinical trials (can run in parallel with step 1)
uv run python scripts/ingest_trials.py

# 3. Extract entities from papers using Claude (resumable — safe to interrupt)
uv run python scripts/extract_entities.py

# 4. Build knowledge graph
uv run python scripts/build_graph.py

# 5. Build ChromaDB vector index
uv run python scripts/build_index.py

# 5.5 Add RAG-grounded mechanism summaries to trials (needs the index from step 5; resumable)
uv run python scripts/ingest_trials.py --summaries

# 6. Build the experimental therapy landscape (offline LLM classification via Batch API)
uv run python scripts/build_landscape.py

# 7. run application
uv run gradio app.py
```

## Key Invariants

- **Node key = `canonical_id`**, never raw entity name. Two papers mentioning "TDP-43" and "TARDBP" must produce one node.
- **ChromaDB metadata values must be scalars** (str/int/float). Lists → comma-separated strings, deserialized on retrieval.
- **KG expansion precedes RAG retrieval** in the agent loop. Never query ChromaDB with the raw user question alone.
- **Trial `mechanism_summary` is RAG-grounded, cite-or-unknown**. `animal_results` and `repurposed_from` are filled only when a retrieved corpus passage supports them (with its PMID); an uncited or unsupported claim is stored as `"unknown"` — never model recall. Built in step 5.5 (needs the index), so it lives in `ingestion/clinicaltrials.py` but runs after `build_index`.
- **All heavy compute is offline**. No PubMed/extraction calls at query time.

## Data File Locations

```
data/papers/papers.jsonl          — 500 ALS paper records (ALSPaper)
data/trials/trials.jsonl          — ALS clinical trial records
data/extracted/entities.jsonl     — per-paper NER output
data/extracted/canonical_ids.json — entity name → canonical ID registry
data/extracted/.progress.json     — extraction resumability tracker
data/graph/als_graph.pkl          — NetworkX DiGraph (fast load)
data/graph/als_graph.json         — human-readable graph export
data/chroma/                      — ChromaDB SQLite store
data/tools/                       — Claude tool input schemas (JSON)
data/seeds/therapy_classes.json   — ALS mechanism taxonomy for the therapy landscape
data/seeds/therapy_gold.json      — gold-labeled therapies for classifier eval/calibration
data/landscape/landscape.json     — experimental therapy landscape (offline-built, committed to git)
```

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | — | Claude API |
| `ENTREZ_EMAIL` | Yes | — | NCBI Entrez (required by NCBI) |
| `NCBI_API_KEY` | No | — | Raises Entrez rate limit 3→10 req/s |
| `LLM_PROVIDER` | No | `anthropic` | Switch to `openai` |
| `CANDLE_LOG_LEVEL` | No | `WARNING` | Console log verbosity |
| `CANDLE_LOG_DIR` | No | `logs` | Log file directory |
