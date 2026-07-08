"""ChromaDB query interface with citation-weighted re-ranking."""
from __future__ import annotations

import math

import chromadb

from config import CHROMA_ENTITY_N_RESULTS, CHROMA_N_RESULTS
from logging_config import get_logger

_logger = get_logger("rag.retriever")


def search(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = CHROMA_N_RESULTS,
) -> list[dict]:
    """
    Semantic search with citation-weighted re-ranking.
    Over-fetches 2× then re-ranks by: similarity * log(citation_count + 2).
    Deduplicates to one chunk per paper (best-scoring chunk wins).
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
    results = _rerank(results)
    return results[:n_results]


def search_by_entities(
    collection: chromadb.Collection,
    entity_names: list[str],
    n_results: int = CHROMA_ENTITY_N_RESULTS,
) -> list[dict]:
    """
    Run one query per entity, merge and deduplicate by PMID.
    Caps at 8 entity queries to avoid excessive API calls.
    """
    if not entity_names or collection.count() == 0:
        return []

    seen: dict[str, dict] = {}
    for entity in entity_names[:8]:
        raw = collection.query(
            query_texts=[entity],
            n_results=min(10, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        for r in _parse_raw(raw):
            pmid = r["pmid"]
            if pmid not in seen or r["score"] > seen[pmid]["score"]:
                seen[pmid] = r

    merged = _rerank(list(seen.values()))
    return merged[:n_results]


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


def _rerank(results: list[dict]) -> list[dict]:
    """
    Apply citation weighting and deduplicate to one chunk per PMID.
    score = similarity * log(citation_count + 2)
    log(2) ≈ 0.69 is the floor for uncited papers, so they're still ranked
    but deprioritized relative to highly-cited work.
    """
    for r in results:
        r["score"] = r["similarity"] * math.log(r["citation_count"] + 2)

    results.sort(key=lambda x: x["score"], reverse=True)

    # Keep best chunk per paper
    seen: dict[str, dict] = {}
    for r in results:
        pmid = r["pmid"]
        if pmid not in seen:
            seen[pmid] = r
    return list(seen.values())
