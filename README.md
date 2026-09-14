---
title: Candle Fire
emoji: 🕯️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.14.0"
app_file: app.py
pinned: false
---

# 🕯️ Candle-Fire

**ALS Research Intelligence for Physicians**

Candle-fire is a physician-facing tool that synthesizes evidence from ~10,000 curated ALS research papers and a biomedical knowledge graph. Ask a free-text question about ALS biology, drug targets, or clinical trials — get a structured, cited answer in under 30 seconds.

## What It Does

- **Two-layer retrieval**: Knowledge graph expansion (BioLORD-2023-C embeddings + NetworkX) → RAG over ~10,000 ALS papers
- **Citation-weighted ranking**: Highly-cited papers surface first
- **Structured synthesis**: Claude Sonnet produces mechanism summaries, entity tables, evidence strength assessments, and trial links
- **Biomedical synonyms**: BioLORD understands that "TDP-43" = "TARDBP" = "TAR DNA-binding protein 43"

## Setup

### Prerequisites

```bash
# Python 3.11+, uv package manager
pip install uv
uv sync
```

### Environment variables

```bash
cp .env.example .env
# Fill in:
#   ANTHROPIC_API_KEY — required
#   ENTREZ_EMAIL      — required for PubMed ingestion
#   NCBI_API_KEY      — optional, raises rate limit 3→10 req/s
```

### Run the offline pipeline (once)

Build the knowledge assets before launching the app. Each step is resumable.

```bash
# 1. Ingest ~10,000 ALS papers from PubMed + PMC full text + citation counts (~15 min)
uv run python scripts/ingest_papers.py

# 2. Ingest ALS clinical trials from ClinicalTrials.gov (< 1 min, run in parallel)
uv run python scripts/ingest_trials.py

# 3. Extract biomedical entities using Claude Sonnet (~$1.50, ~50 min, resumable)
uv run python scripts/extract_entities.py

# 4. Build the knowledge graph (~5 sec)
uv run python scripts/build_graph.py

# 5. Build the ChromaDB vector index with BioLORD embeddings (~10 min, one-time model download)
uv run python scripts/build_index.py
```

### Launch

```bash
# Web UI (auto-reloads on file changes — use during development)
uv run gradio app.py

# Web UI (production / one-shot)
uv run python app.py

# CLI
uv run python main.py "What is the evidence for tofersen targeting SOD1?"
```

## Deploy to Hugging Face

Deployment uses **two repos**, because runtime data is kept separate from code:

| Repo | Holds | Shipped by |
|---|---|---|
| Space `KevinIsCoding/candle-fire` (git remote `hf`) | **Code** + small committed data (`landscape.json`, seeds, tool schemas) | `git push hf main` |
| Dataset `KevinIsCoding/candle-fire-data` | **Large runtime data** — ChromaDB index, graph pickle, `trials.jsonl` (all gitignored) | `scripts/upload_data.py` |

The large data is gitignored (see `.gitignore`) and never travels with the code. On startup the Space pulls it from the dataset repo via `app._ensure_data()`. The Space has **no persistent storage**, so every restart is a clean cold start that re-downloads this data — which means the dataset repo must always hold the current files.

**Deploy in this order (data first, then code):**

```bash
# 0. Merge the approved PR into main and sync locally
git checkout main && git pull origin main

# 1. Push runtime data to the dataset repo (REQUIRED before the code push)
#    Needs a HF token with write access: `huggingface-cli login` or HF_TOKEN env var.
uv run python scripts/upload_data.py            # uploads chroma/, graph, trials.jsonl
uv run python scripts/upload_data.py --dry-run  # preview targets + sizes, no writes
uv run python scripts/upload_data.py --only trials   # push just one target

# 2. Deploy the code (triggers the Space rebuild)
git push hf main
```

> **Why data first:** the Space cold-starts on rebuild and immediately fetches data from the
> dataset repo. If you push code before uploading data, the Space can boot against stale or
> missing files — e.g. a missing `trials.jsonl` silently disables trial search. Always run
> `upload_data.py` before `git push hf main`. The upload/download target is a single constant
> (`config.HF_DATASET_REPO`) shared by the script and the app, so they can't drift.

## Architecture

```
Physician query
  → agents/research_agent.py
      1. Claude: extract query entities → ["SOD1", "tofersen"]
      2. graph/query.py: KG expansion → ["SOD1", "TARDBP", "antisense oligonucleotide", ...]
      3. rag/retriever.py: BioLORD semantic search + entity search → top 15 papers
         (re-ranked by: similarity × log(citation_count + 2))
      4. graph/query.py: find linked clinical trials
      5. Claude Sonnet (streaming): synthesize research landscape
  → Gradio UI (streaming response with citations)
```

**Embedding model**: `FremyCompany/BioLORD-2023-C` — anchored to UMLS/SNOMED CT/MeSH ontologies, natively resolves biomedical synonyms.

**Knowledge graph**: NetworkX DiGraph with Gene/Protein/Compound/Pathway/Phenotype/Mechanism/ClinicalTrial nodes. 1-hop BFS expansion before RAG retrieval.

## Data Sources

| Source | Content | Volume |
|---|---|---|
| PubMed Entrez | ALS paper abstracts + metadata | ~10,000 papers |
| PubMed Central | Full text for Open Access papers | ~50% coverage |
| Semantic Scholar | Citation counts per paper | All papers |
| ClinicalTrials.gov v2 | Interventional + expanded-access ALS trials | ~720 trials |

## Disclaimer

Research synthesis tool. Always verify claims with primary sources before applying to patient care. Not a substitute for clinical judgment.
