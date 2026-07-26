# Design Doc: candle-fire Vector DB — Deployment Fix & Migration Path

> **Status:** Draft for review · **Author:** _<you>_ · **Reviewers:** _<add>_ · **Date:** 2026-07
> This is a design doc to review/modify **before** implementation. It ends with open
> decisions (Section 10) you should resolve; the recommendation is phased so Phase 0 can
> proceed independently of the Phase 1 decision.

## 1. Context & Problem

candle-fire's RAG layer uses an **embedded ChromaDB (SQLite)** vector index. The concern
raised: the vector DB "keeps getting larger" — is it time to migrate (e.g., to AWS)?

**Current numbers (measured):**
- **30,967 vectors** (chunks) from ~10k papers; **1.2 GB** on disk (`data/chroma/chroma.sqlite3`).
- Corpus capped at **20k papers** (`config.PUBMED_DEFAULT_MAX`) → ~62k chunks / ~2.4 GB at the cap.
- Embeddings: **BioLORD-2023-C** (768-dim). Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Deployment:** the index is stored in the HF **dataset** repo (`KevinIsCoding/candle-fire-data`)
  and **downloaded to the Space at cold start** (`app.py:_ensure_data`). The 1.2 GB download
  amplified the recent 503 outage.

## 2. Key Finding — this is a *deployment* problem, not a *scale* problem

- **31k–62k vectors is trivially small** for any vector store (they scale to millions/billions).
  Chroma is not being outgrown in capability.
- **The 1.2 GB is mostly text, not vectors.** Vectors alone are ~100 MB (31k × 768 × 4 bytes).
  The rest is full document text + a full-text index — because the retriever relies on Chroma's
  substring search (`where_document $contains`) to ground drug codes / gene IDs. That text is
  **load-bearing** for citation grounding; it can't just be dropped to shrink the index.
- Therefore the pain is **where the DB lives** (bundled + downloaded per cold start), not its size.

## 3. Goals / Non-Goals

**Goals**
- Remove the 1.2 GB cold-start download that couples the DB to the Space lifecycle.
- Preserve the **hybrid retrieval** (semantic ANN + exact substring + metadata filter) — load-bearing for grounding.
- Keep a low-risk path that also scales if the corpus grows.

**Non-Goals**
- Shrinking the index by dropping document text (breaks grounding).
- Rewriting the ranking pipeline (RRF, cross-encoder, citation/recency boost) — it is backend-independent and stays as-is.

## 4. Current Architecture (what any migration must preserve)

**Retriever (`rag/retriever.py`)** splits cleanly into two layers:

- **Candidate fetch — backend-specific (the only code that must change on migration):**
  - `search()` / `search_by_entities()`: semantic ANN via `collection.query(query_texts=...)`.
  - `search_by_keyword()`, `is_grounded_in_corpus()`, `is_grounded_in_abstract()`: **exact substring**
    via `where_document={"$contains": ...}`, expanded over `_term_variants()` (SPG302 / SPG 302 / SPG-302).
  - `get_paper()`, `paper_texts_for_pmids()`: metadata fetch via `collection.get(where=...)`.
- **Ranking — backend-independent (REUSE UNCHANGED):**
  - `_parse_raw()` normalizes any backend response to the result-dict shape.
  - `rrf_merge()`, `cross_encoder_rerank()`, `apply_citation_boost()` operate purely on result dicts.

**Indexer (`rag/indexer.py`)**: `build_collection()` chunks papers (≤6 chunks/paper), embeds via a
`SentenceTransformerEmbeddingFunction` **baked into the Chroma collection**, IDs `{pmid}_s{i}`, scalarized metadata.

> **Key design lever:** only the candidate-fetch functions touch the backend. If a new backend keeps
> the `_parse_raw` output shape, the entire ranking/fusion/boost pipeline is reused verbatim. This
> bounds migration scope to ~6 functions + the indexer writer.

## 5. Options Considered

| Option | Fit | Verdict |
|---|---|---|
| **A. Cheap deploy fix** (persistent storage on the Space, and/or Chroma client/server on a small host) | Keeps all Chroma features; ~no code change | **Recommended first (Phase 0)** |
| **B. Postgres + pgvector** (AWS Aurora/RDS, or Neon/Supabase) | One store for vectors (`<=>`) + full-text (ILIKE/tsvector) + metadata (SQL); preserves the hybrid | **Recommended migration target (Phase 1)** |
| **C. AWS OpenSearch** (kNN + BM25 + filters) | Fits the hybrid; AWS-native | Viable but heavier ops/tuning; overkill at 31k vectors |
| **D. Vector-only managed** (Pinecone / Qdrant Cloud) | No native substring/full-text | **Rejected** — would require a separate keyword store to keep grounding |
| **E. Amazon Bedrock Knowledge Bases** | Managed end-to-end RAG | **Rejected** — replaces the custom hybrid + grounding + citation-weighting (the product's differentiators) |

## 6. Recommendation (phased)

- **Phase 0 — now (do regardless of the migration decision):** Option A. Enable **persistent storage**
  on the Space so `_ensure_data` stops re-downloading 1.2 GB each restart; optionally run **Chroma in
  client/server mode** so the Space queries it over HTTP. Removes the cold-start pain with little/no
  rewrite. (Pairs with the separate COE action item to make data refresh not require a factory-reboot.)
- **Phase 1 — when a trigger hits:** Option B (pgvector). **Triggers:** corpus scaled toward 100k+
  chunks / continuous ingestion; OR a need to decouple the DB from the Space (multi-instance, independent
  updates); OR wanting to stop shipping a data blob entirely. Abstract the backend first (Section 7) so
  the swap is localized and reversible.

## 7. Phase 1 Detailed Design (pgvector)

**Backend abstraction (do this even before migrating):** introduce `rag/backends/` with a thin interface —
`semantic_candidates(query, n)`, `substring_candidates(term_variants, n)`, `fetch_by_pmids(pmids)`,
`abstract_contains(variant)`, `count()` — implemented by `chroma_backend` (wraps today's code) and
`pg_backend`. `retriever.py` calls the interface; the ranking layer is untouched. `indexer.py` gains a pg writer.

**Schema (single table):**
```sql
CREATE TABLE chunks (
  chunk_id       text PRIMARY KEY,
  pmid           text,
  section        text,
  chunk_index    int,
  title          text,
  year           int,
  doi            text,
  citation_count int,
  entity_names   text,
  has_full_text  boolean,
  document       text,
  embedding      vector(768),
  doc_tsv        tsvector
);
-- indexes: HNSW on embedding (vector_cosine_ops); GIN on doc_tsv; btree on pmid
```

**Query mapping (preserve current semantics):**
- `collection.query(query_texts)` → embed query with BioLORD explicitly, then
  `SELECT ... ORDER BY embedding <=> $qvec LIMIT n`.
- `where_document $contains` → `WHERE document ILIKE '%'||$variant||'%'` (exact substring — matches
  Chroma `$contains` behavior for drug codes; **use ILIKE, not tsvector**, to keep grounding parity).
- `where={pmid $in}` / `chunk_index` / `get` → plain SQL `WHERE`.
- Query embeddings use the same `config` model; the ranking pipeline consumes `_parse_raw`-shaped dicts unchanged.

**Config:** DB URL via env var; embedding model unchanged. Graph/agent layers untouched.

## 8. Rollout / Migration

1. Stand up Postgres + pgvector (RDS/Aurora/Neon); `CREATE EXTENSION vector`.
2. **Backfill** with a one-off script that reuses the existing `rag/indexer.py` chunker on `papers.jsonl`,
   embeds, and writes rows (idempotent on `chunk_id`).
3. **Dual-read validation** (Section 11) — compare Chroma vs pgvector on a fixed query set.
4. Flip the Space to the pg backend via an **env flag**, keeping the Chroma path behind the flag for rollback.
5. Once stable, drop the 1.2 GB blob from the dataset repo.

## 9. Risks & Mitigations

- **Substring-grounding parity:** ILIKE must reproduce `$contains` over `_term_variants()`. Port the variant
  expansion; add tests comparing grounding booleans and keyword hits against Chroma.
- **Per-request query fan-out:** `search_by_entities()` issues up to 12 queries/request — batch into a single
  SQL round-trip or reduce `RETRIEVAL_ENTITY_QUERY_CAP`; HNSW keeps ANN fast.
- **DB cost/ops:** smallest managed tier is ample at this scale.
- **Embedding drift:** pin the embedding model; re-embed if it changes.

## 10. Open Questions / Decisions for Reviewer

1. **Growth:** staying ~10–20k papers, or scaling toward full PubMed / continuous ingestion? (Decides whether Phase 1 is needed at all.)
2. **Host:** AWS specifically (Aurora/RDS), or is managed-elsewhere (Neon/Supabase) acceptable — often cheaper/simpler at this scale?
3. **Driver:** is decoupling the DB from the Space a goal in itself, or is cold-start relief (Phase 0) enough for now?
4. **Ops/budget:** appetite for an always-on DB instance vs the current zero-infra file model.

## 11. Verification

- **Parity harness:** fixed query set; assert the final top-15 cited PMIDs and the grounding booleans
  (`is_grounded_in_corpus`, `is_grounded_in_abstract`) match Chroma within tolerance.
- **Latency:** p50/p95 per query under both backends.
- **Cold-start:** Space boot time before/after — Phase 0 target: no 1.2 GB re-download; Phase 1 target: no data blob shipped at all.
