"""ChromaDB query interface with RRF merge and cross-encoder reranking."""
from __future__ import annotations

import math
import re

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


def _term_variants(term: str) -> list[str]:
    """
    Generate letter/digit boundary variants of a compound identifier so that
    "SPG302", "SPG 302", and "SPG-302" all resolve to the same papers.
    "SPG302" → ["SPG302", "SPG 302", "SPG-302"]
    """
    # Collapse any existing space/hyphen at letter–digit boundaries → canonical form
    canonical = re.sub(r'(?<=[A-Za-z])[\s\-](?=\d)|(?<=\d)[\s\-](?=[A-Za-z])', '', term)
    spaced = re.sub(r'([A-Za-z])(\d)', r'\1 \2', canonical)
    hyphenated = re.sub(r'([A-Za-z])(\d)', r'\1-\2', canonical)
    return list(dict.fromkeys([term, canonical, spaced, hyphenated]))


def term_matches_text(text: str, term: str) -> bool:
    """True if any spacing/hyphen variant of `term` appears in `text` (case-insensitive)."""
    if not text or not term.strip():
        return False
    low = text.lower()
    return any(v.lower() in low for v in _term_variants(term) if v.strip())


def paper_texts_for_pmids(
    collection: chromadb.Collection,
    pmids: list[str],
) -> dict[str, dict[str, str]]:
    """
    Fetch ALL chunks for each PMID in one call and return, per PMID:
      {"abstract": <chunk_index 0 doc>, "full": <all chunk docs concatenated>}.
    Used to distinguish "the paper is actually about this compound" (present in the
    abstract) from an incidental full-text-only mention (present somewhere in the
    body, e.g. a drug-pipeline table, but not the abstract). Fetching every chunk is
    required because the retrieved representative chunk is often NOT the one holding
    the compound name.
    """
    pmids = [p for p in dict.fromkeys(pmids) if p]
    if not pmids:
        return {}
    try:
        res = collection.get(
            where={"pmid": {"$in": pmids}},
            include=["documents", "metadatas"],
        )
    except Exception:
        return {}
    out: dict[str, dict[str, str]] = {p: {"abstract": "", "full": ""} for p in pmids}
    parts: dict[str, list[str]] = {p: [] for p in pmids}
    for meta, doc in zip(res.get("metadatas", []), res.get("documents", [])):
        pmid = meta.get("pmid", "")
        if pmid not in out:
            continue
        doc = doc or ""
        parts[pmid].append(doc)
        if meta.get("chunk_index") == 0:
            out[pmid]["abstract"] = doc
    for pmid in out:
        out[pmid]["full"] = "\n".join(parts[pmid])
    return out


def is_grounded_in_abstract(collection: chromadb.Collection, term: str) -> bool:
    """
    True if any variant of `term` appears in an abstract chunk (chunk_index == 0).
    Signals that at least one paper is genuinely *about* the term, as opposed to
    only naming it in a full-text pipeline/landscape table.
    """
    if not term.strip() or collection.count() == 0:
        return False
    for variant in _term_variants(term):
        if not variant.strip():
            continue
        try:
            raw = collection.get(
                where={"chunk_index": {"$eq": 0}},
                where_document={"$contains": variant},
                limit=1,
                include=["metadatas"],
            )
        except Exception:
            continue
        if raw.get("ids"):
            return True
    return False


def is_grounded_in_corpus(collection: chromadb.Collection, term: str) -> bool:
    """
    True if any spacing/hyphen variant of `term` appears literally in a paper
    (ChromaDB $contains). This is exact-substring presence — the reliable signal
    for "is this named entity actually written in the corpus", as opposed to
    semantic search which always returns nearest neighbors regardless of relevance.
    """
    if not term.strip() or collection.count() == 0:
        return False
    for variant in _term_variants(term):
        if not variant.strip():
            continue
        try:
            raw = collection.query(
                query_texts=[variant],
                where_document={"$contains": variant},
                n_results=1,
                include=["metadatas"],
            )
        except Exception:
            continue
        if raw.get("ids", [[]])[0]:
            return True
    return False


def search_by_keyword(
    collection: chromadb.Collection,
    terms: list[str],
    n_results: int = 10,
) -> list[dict]:
    """
    Exact-substring search using ChromaDB's where_document $contains filter.
    Searches all spacing/hyphen variants of each term so "SPG302", "SPG 302",
    and "SPG-302" all resolve to the same papers. Catches proper nouns (drug
    codes, gene IDs) whose embeddings are meaningless to the model.
    """
    if not terms or collection.count() == 0:
        return []

    seen: dict[str, dict] = {}
    for term in terms:
        if not term.strip():
            continue
        for variant in _term_variants(term):
            if not variant.strip():
                continue
            try:
                raw = collection.query(
                    query_texts=[variant],
                    where_document={"$contains": variant},
                    n_results=min(n_results, collection.count()),
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                continue
            for r in _parse_raw(raw):
                pmid = r["pmid"]
                if pmid not in seen or r["similarity"] > seen[pmid]["similarity"]:
                    seen[pmid] = r

    return _dedup_by_pmid(list(seen.values()))


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


_RECENCY_BASE_YEAR = 1990
_RECENCY_CURRENT_YEAR = 2025
_RECENCY_MAX_BOOST = 0.5  # most recent papers get 1.5× vs oldest at 1.0×


def apply_citation_boost(results: list[dict]) -> list[dict]:
    """
    Final score = ce_score × log(citation_count + 2) × recency_factor.

    Citation factor: log-scaled so each order-of-magnitude in citations adds
    roughly equal weight. log(2) ≈ 0.69 floor for uncited papers.

    Recency factor: linear 1.0 → 1.5 from 1990 to 2025. A 2024 paper scores
    50% higher than a 1990 paper at equal citation count and relevance, reflecting
    that recent evidence is more likely to reflect current understanding.
    """
    for r in results:
        base = r.get("ce_score", r.get("similarity", 0.0))
        citation_factor = math.log(r["citation_count"] + 2)
        year = r.get("year") or _RECENCY_BASE_YEAR
        recency_factor = 1.0 + _RECENCY_MAX_BOOST * (
            max(0, year - _RECENCY_BASE_YEAR)
            / (_RECENCY_CURRENT_YEAR - _RECENCY_BASE_YEAR)
        )
        r["score"] = base * citation_factor * recency_factor
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
