# Candle-Fire: ALS Research Landscape Tool

## Project Initiative

Candle-fire is a physician-facing ALS research intelligence platform — a sibling to beacon. Where beacon helps patients find clinical trials, candle-fire helps physicians understand the *research evidence behind* those trials and ALS biology more broadly.

**The problem**: A physician encounters a clinical trial for an antisense oligonucleotide targeting SOD1. To evaluate it, they need to know: What is the evidence base for SOD1 as a target? What are the known mechanisms? What else has been tried? Today this requires hours of manual literature review.

**The solution**: A physician types a free-text question ("What's the evidence for tofersen targeting SOD1 in ALS?") and gets a synthesized, cited answer grounded in ~500 curated ALS papers, enriched by a knowledge graph linking genes, proteins, compounds, pathways, and clinical trials — delivered in under 30 seconds.

**End state**: A Gradio web app deployable on HuggingFace Spaces. Two layers of intelligence: (1) a vector RAG layer for semantic paper retrieval, (2) a knowledge graph layer for entity-level relationship traversal. Claude Sonnet synthesizes both into a structured research landscape report with citations, mechanism summaries, and related trial links.

---

## Critique of Original Plan

1. **RAG alone is semantically blind.** Query for "riluzole" must surface "glutamate excitotoxicity" via the KG (riluzole → INHIBITS → glutamate excitotoxicity). KG expansion *precedes* RAG retrieval — not optional.

2. **Entity normalization is a prerequisite for KG construction.** TDP-43 / TARDBP / TDP43 must resolve to one canonical node before graph build. The normalizer runs as part of extraction and writes `canonical_ids.json` as the single source of truth.

3. **All heavy compute is offline batch.** Ingestion, extraction, KG build, RAG indexing — run once. Query time: NetworkX pickle (read) + ChromaDB (read) + Anthropic API only.

4. **Citation counts weight evidence quality.** A paper cited 500 times is stronger evidence than one cited 5 times. Fetch citation counts from Semantic Scholar API (free, no auth). Use to re-rank RAG results and weight KG edge confidence.

5. **PMC XML full text, not raw PDFs.** ~50-60% of recent ALS papers are PMC Open Access. Ingest structured XML (section-labeled: Introduction, Methods, Results, Discussion) via PubMed Entrez. Always fall back to abstract. Avoid PDF parsing — too brittle and legally ambiguous for paywalled papers.

---

## Architecture

### Tech Stack
- **Language**: Python 3.11+, uv package manager
- **LLM**: Claude Sonnet 4-6 (entity extraction + research synthesis)
- **Vector store**: ChromaDB (SQLite-backed, persistent, no separate service)
- **Graph**: NetworkX DiGraph (pickled, loaded once at startup, scales to 20K+ papers)
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers (local, no API needed)
- **UI**: Gradio 6 (HuggingFace Spaces deployment, mirrors beacon)
- **Paper source**: PubMed Entrez API + PMC XML full text (hybrid)
- **Citation counts**: Semantic Scholar API (free, by DOI/PMID)

### Data Flow

```
OFFLINE PIPELINE (run once, in order):

  Stage 1: Ingestion
    scripts/ingest_papers.py
      → ingestion/pubmed.py (PubMed Entrez, batch 200 PMIDs, abstract always)
      → ingestion/pmc.py (PMC XML full text for OA papers, ~50-60% coverage)
      → ingestion/semantic_scholar.py (citation counts per PMID/DOI)
      → data/papers/papers.jsonl  [ALSPaper: title, abstract, full_text?, citation_count, ...]

    scripts/ingest_trials.py  [parallel with above]
      → ingestion/clinicaltrials.py (ClinicalTrials.gov v2, condition=ALS)
      → data/trials/trials.jsonl

  Stage 2: Entity Extraction
    scripts/extract_entities.py
      → extraction/extractor.py (Claude Sonnet, 10 papers/call, retry+backoff, resumable)
      → extraction/normalizer.py (HGNC alias table + PubChem + HGNC REST fallback)
      → data/extracted/entities.jsonl
      → data/extracted/canonical_ids.json

  Stage 3: Knowledge Graph Build
    scripts/build_graph.py
      → graph/builder.py (NetworkX DiGraph, upsert nodes+edges, weight by citation_count)
      → data/graph/als_graph.pkl + als_graph.json

  Stage 4: RAG Index Build
    scripts/build_index.py
      → rag/indexer.py (ChromaDB "als_papers", chunk by section if full text, else abstract)
      → data/chroma/

ONLINE QUERY PIPELINE (per physician request):

  Physician free-text question
    → agents/research_agent.py
        1. Claude: extract query entities → ["SOD1", "antisense oligonucleotide"]
        2. graph/query.py: 1-hop KG expansion → ["SOD1", "TARDBP", "tofersen", "RNA splicing", ...]
        3. rag/retriever.py: semantic search + entity-filtered search → top 15 papers
           (re-ranked by: ChromaDB distance × log(citation_count + 1))
        4. graph/query.py: find_trials_for_target() for each query entity
        5. Claude Sonnet (streaming): synthesize ResearchLandscape
    → Gradio UI (streaming response with citations)
```

---

## File Structure

```
candle-fire/
├── pyproject.toml
├── .env.example             # ANTHROPIC_API_KEY, ENTREZ_EMAIL, NCBI_API_KEY
├── CLAUDE.md                # Architecture guide (module responsibilities)
│
├── config.py                # Model names, paths, ALS seed entities, API endpoints
├── models.py                # Dataclasses: ALSPaper, ExtractedEntity, EntityRelationship, ResearchLandscape
├── prompts.py               # System prompts: extraction + synthesis
├── tools.py                 # Tool schema loader (mirrors beacon/tools.py)
├── llm.py                   # LLM provider abstraction (copied from beacon/llm.py)
├── logging_config.py        # Structured JSON logging (adapted from beacon/beacon_logging.py)
├── app.py                   # Gradio UI entry point (graph + ChromaDB loaded at startup)
├── main.py                  # CLI entry point (Rich console)
│
├── ingestion/
│   ├── pubmed.py            # PubMed Entrez: fetch abstracts + metadata by query or PMID list
│   ├── pmc.py               # PMC XML full text: fetch & parse structured sections for OA papers
│   ├── clinicaltrials.py    # ClinicalTrials.gov v2 (adapted from beacon/trials_api.py, no geo)
│   └── semantic_scholar.py  # Citation counts by PMID/DOI (batch API, free tier)
│
├── extraction/
│   ├── extractor.py         # Claude Sonnet NER: batch 10 papers, retry+backoff, .progress.json
│   └── normalizer.py        # Canonical ID mapping (HGNC alias table + PubChem + REST fallback)
│
├── graph/
│   ├── builder.py           # NetworkX DiGraph: upsert nodes+edges, seed ALS entities, citation weighting
│   ├── query.py             # Traversal: expand_query_entities, find_trials_for_target, get_entity_evidence
│   └── serializer.py        # Save/load: pickle (fast) + JSON (human-readable)
│
├── rag/
│   ├── indexer.py           # ChromaDB collection builder: section-aware chunking, citation_count metadata
│   └── retriever.py         # search(), search_by_entities(), citation-weighted re-ranking
│
├── agents/
│   └── research_agent.py    # Multi-step synthesis agent (streaming, mirrors beacon/agents/research.py)
│
├── scripts/
│   ├── ingest_papers.py     # CLI: PubMed + PMC XML + citation counts → papers.jsonl
│   ├── ingest_trials.py     # CLI: ClinicalTrials.gov → trials.jsonl
│   ├── extract_entities.py  # CLI: papers.jsonl → entities.jsonl (resumable)
│   ├── build_graph.py       # CLI: entities.jsonl + trials.jsonl → als_graph.pkl
│   └── build_index.py       # CLI: papers.jsonl + entities.jsonl → data/chroma/
│
├── data/
│   ├── papers/papers.jsonl
│   ├── trials/trials.jsonl
│   ├── extracted/
│   │   ├── entities.jsonl
│   │   ├── canonical_ids.json
│   │   └── .progress.json   # Extraction resumability tracker
│   ├── graph/
│   │   ├── als_graph.pkl
│   │   └── als_graph.json
│   ├── chroma/              # ChromaDB SQLite store
│   └── tools/
│       ├── extract_entities.json
│       └── search_landscape.json
│
└── tests/
    ├── test_pubmed.py
    ├── test_pmc.py
    ├── test_semantic_scholar.py
    ├── test_extractor.py
    ├── test_normalizer.py
    ├── test_graph_builder.py
    ├── test_graph_query.py
    ├── test_retriever.py
    └── test_research_agent.py
```

---

## Staged Implementation Plan

### Stage 1 — Project Foundation
**Goal**: Runnable skeleton with all dependencies wired.

- `pyproject.toml` with all dependencies (anthropic, gradio, chromadb, networkx, biopython, httpx, sentence-transformers, rich, python-dotenv)
- `config.py`, `models.py`, `prompts.py` (stubs), `tools.py`, `llm.py` (copied from beacon), `logging_config.py`
- `data/tools/extract_entities.json` and `search_landscape.json` tool schemas
- All `__init__.py` files, `.env.example`, `CLAUDE.md`

**Done when**: `uv run python -c "import anthropic, chromadb, networkx, Bio"` passes with no errors.

---

### Stage 2 — Paper Ingestion Pipeline
**Goal**: Populate `data/papers/papers.jsonl` with ~500 ALS papers including citation counts.

- `ingestion/pubmed.py`: PubMed Entrez client (fetch by MeSH query or PMID file, batch 200)
- `ingestion/pmc.py`: PMC XML full-text fetcher for OA papers (parse by section)
- `ingestion/semantic_scholar.py`: Citation count enrichment (batch by PMID)
- `scripts/ingest_papers.py`: Orchestrates all three, writes `papers.jsonl`
- `scripts/ingest_trials.py` + `ingestion/clinicaltrials.py`: ALS trials → `trials.jsonl`

PubMed seed query: `"amyotrophic lateral sclerosis"[MeSH Major Topic] AND ("2018"[PDAT]:"2024"[PDAT]) AND hasabstract[text]`

**Done when**: `papers.jsonl` has 500 lines with `citation_count` populated; `trials.jsonl` has 20+ records.

---

### Stage 3 — RAG Pipeline + Working v0
**Goal**: End-to-end working query pipeline using RAG only (no KG yet).

- `rag/indexer.py`: ChromaDB collection builder (section-aware chunking, citation_count in metadata)
- `rag/retriever.py`: `search()`, `search_by_entities()`, citation-weighted re-ranking
- `scripts/build_index.py`
- `agents/research_agent.py` (RAG-only version, no KG expansion step yet)
- `prompts.py` synthesis prompt
- `main.py`: CLI interface

**Done when**: `uv run python main.py "What compounds target glutamate excitotoxicity in ALS?"` returns a synthesized answer with cited PMIDs.

---

### Stage 4 — Entity Extraction + Knowledge Graph
**Goal**: Offline pipeline produces a populated KG; query agent upgrades to KG+RAG.

- `extraction/extractor.py`: Claude Sonnet NER, 10 papers/call, resumable via `.progress.json`
- `extraction/normalizer.py`: HGNC alias table (~50 ALS genes) + PubChem fallback + REST fallback
- `scripts/extract_entities.py`
- `graph/builder.py`: NetworkX DiGraph with upsert, citation-weighted edge confidence, seed entities
- `graph/query.py`: `expand_query_entities()`, `find_trials_for_target()`, `get_entity_evidence()`
- `graph/serializer.py`
- `scripts/build_graph.py`
- Upgrade `agents/research_agent.py` to include KG expansion step before RAG

**Done when**: `G.number_of_nodes() > 100`; query for "tofersen" surfaces SOD1 as the mechanism link (not just literal tofersen matches).

---

### Stage 5 — Gradio UI + Deployment Polish
**Goal**: Browser-accessible web app ready for HuggingFace Spaces.

- `app.py`: Gradio UI with streaming, example questions sidebar, citation display, disclaimer
- Module-level graph + ChromaDB client initialization (once at startup)
- `requirements.txt` (HF Spaces mirror of pyproject.toml)
- `README.md`: setup instructions, offline pipeline run order, env vars
- Tests: `uv run pytest` all pass

**Done when**: `uv run gradio app.py` → physician asks "What's the mechanism of tofersen in ALS?" → streaming response includes SOD1 mechanism, 3+ cited papers, at least 1 NCT trial link, and a disclaimer.

---

## Key Reuse from Beacon

| Beacon file | Candle-fire file | Change |
|---|---|---|
| `beacon/llm.py` | `llm.py` | Copy verbatim; model constants from `config.py` |
| `beacon/beacon_logging.py` | `logging_config.py` | Namespace → `candle_fire` |
| `beacon/trials_api.py` | `ingestion/clinicaltrials.py` | Remove geo/distance; add `extract_target_entities()` |
| `beacon/tools.py` | `tools.py` | Copy pattern; update tool names |
| `beacon/agents/research.py` | `agents/research_agent.py` | Adapt streaming loop; replace tool handlers |
| `beacon/app.py` | `app.py` | Single-turn instead of multi-turn state machine |

---

## Estimated Costs

| Item | Cost |
|---|---|
| Entity extraction (500 papers, 50 Sonnet calls ~10K tokens each) | ~$1.50 one-time |
| Semantic Scholar citation counts | Free |
| PMC XML full text | Free |
| Per physician query (Sonnet, ~15K tokens in+out) | ~$0.05/query |
