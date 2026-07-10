"""ChromaDB query interface with RRF merge and cross-encoder reranking."""
from __future__ import annotations

import math

import chromadb

from config import (
    CHROMA_ENTITY_N_RESULTS,
    CHROMA_N_RESULTS,
    CROSS_ENCODER_TOP_N,
    RETRIEVAL_ENTITY_N,
    RETRIEVAL_ENTITY_QUERY_CAP,
    RETRIEVAL_SEMANTIC_N,
    RRF_K,
    RRF_TOP_N,
)
from logging_config import get_logger

_logger = get_logger("rag.retriever")


def search(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = RETRIEVAL_SEMANTIC_N,
) -> list[dict]:
    """
    Semantic search — returns pure similarity-ranked results (no citation weighting).
    Over-fetches 2× then deduplicates to one chunk per paper.
    Citation boost is applied downstream after cross-encoder reranking.
    """
    n_fetch = min(n_results * 2, collection.count())
    if n_fetch == 0:
        return []

    raw = collection.query(
        query_texts=[query_text],
        n_results=n_fetch,
        include=["documents", "metadatas", "distances"],
    )
    results = _parse_raw(raw)
    results = _dedup_by_pmid(results)
    return results[:n_results]


def search_by_entities(
    collection: chromadb.Collection,
    entity_names: list[str],
    n_results: int = RETRIEVAL_ENTITY_N,
) -> list[dict]:
    """
    Run one ChromaDB query per entity name, merge and deduplicate by PMID.
    Caps at RETRIEVAL_ENTITY_QUERY_CAP (12) entity queries to bound latency.
    """
    if not entity_names or collection.count() == 0:
        return []

    seen: dict[str, dict] = {}
    for entity in entity_names[:RETRIEVAL_ENTITY_QUERY_CAP]:
        raw = collection.query(
            query_texts=[entity],
            n_results=min(10, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        for r in _parse_raw(raw):
            pmid = r["pmid"]
            if pmid not in seen or r["similarity"] > seen[pmid]["similarity"]:
                seen[pmid] = r

    merged = _dedup_by_pmid(list(seen.values()))
    return merged[:n_results]


def rrf_merge(
    ranked_lists: list[list[dict]],
    k: int = RRF_K,
    top_n: int = RRF_TOP_N,
) -> list[dict]:
    """
    Reciprocal Rank Fusion — combines N ranked lists into one.
    score(pmid) = Σ 1 / (k + rank_in_list_i + 1)
    Preserves the best-scoring dict per PMID from all input lists.
    """
    rrf_scores: dict[str, float] = {}
    best: dict[str, dict] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked):
            pmid = result["pmid"]
            rrf_scores[pmid] = rrf_scores.get(pmid, 0.0) + 1.0 / (k + rank + 1)
            if pmid not in best or result["similarity"] > best[pmid]["similarity"]:
                best[pmid] = result

    sorted_pmids = sorted(rrf_scores, key=lambda p: rrf_scores[p], reverse=True)
    merged = []
    for pmid in sorted_pmids[:top_n]:
        r = best[pmid].copy()
        r["rrf_score"] = round(rrf_scores[pmid], 6)
        merged.append(r)
    return merged


def cross_encoder_rerank(
    model,
    query: str,
    candidates: list[dict],
    top_n: int = CROSS_ENCODER_TOP_N,
) -> list[dict]:
    """
    Cross-encoder reranking — scores (query, document) pairs jointly.
    Truncates document text to 1800 chars (~450 tokens) so query+doc fits
    within the ms-marco model's 512-token limit.
    """
    if not candidates:
        return []

    pairs = [(query, r["document"][:1800]) for r in candidates]
    ce_scores = model.predict(pairs, show_progress_bar=False)

    for r, score in zip(candidates, ce_scores):
        r["ce_score"] = float(score)

    candidates.sort(key=lambda x: x["ce_score"], reverse=True)
    return candidates[:top_n]


def apply_citation_boost(results: list[dict]) -> list[dict]:
    """
    Final score = cross_encoder_score × log(citation_count + 2).
    Applied after cross-encoder so citation quality amplifies — not corrupts — relevance.
    log(2) ≈ 0.69 is the floor for uncited papers.
    """
    for r in results:
        base = r.get("ce_score", r.get("similarity", 0.0))
        r["score"] = base * math.log(r["citation_count"] + 2)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def get_paper(collection: chromadb.Collection, pmid: str) -> dict | None:
    """Retrieve a specific paper's abstract chunk by PMID."""
    result = collection.get(
        where={"$and": [{"pmid": {"$eq": pmid}}, {"chunk_index": {"$eq": 0}}]},
        include=["documents", "metadatas"],
    )
    ids = result.get("ids", [])
    if not ids:
        return None
    meta = result["metadatas"][0]
    return {
        "pmid": pmid,
        "title": meta.get("title", ""),
        "year": meta.get("year", 0),
        "doi": meta.get("doi", ""),
        "citation_count": meta.get("citation_count", 0),
        "document": result["documents"][0],
    }


def _parse_raw(raw: dict) -> list[dict]:
    """Flatten a ChromaDB query response into a list of result dicts."""
    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    results = []
    for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
        similarity = max(0.0, 1.0 - dist)
        citation_count = int(meta.get("citation_count", 0))
        results.append({
            "chunk_id": chunk_id,
            "pmid": meta.get("pmid", ""),
            "title": meta.get("title", ""),
            "year": int(meta.get("year", 0)),
            "doi": meta.get("doi", ""),
            "section": meta.get("section", "abstract"),
            "citation_count": citation_count,
            "entity_names": [e for e in meta.get("entity_names", "").split(",") if e],
            "has_full_text": bool(meta.get("has_full_text", 0)),
            "document": doc,
            "similarity": similarity,
            "score": similarity,
        })
    return results


def _dedup_by_pmid(results: list[dict]) -> list[dict]:
    """Keep best-similarity chunk per paper, sorted by similarity descending."""
    results.sort(key=lambda x: x["similarity"], reverse=True)
    seen: dict[str, dict] = {}
    for r in results:
        pmid = r["pmid"]
        if pmid not in seen:
            seen[pmid] = r
    return list(seen.values())
