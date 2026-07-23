"""Semantic Scholar API client for citation count enrichment."""
from __future__ import annotations

import time

import httpx

from config import SEMANTIC_SCHOLAR_BASE
from logging_config import get_logger

_logger = get_logger("ingestion.semantic_scholar")

_BATCH_SIZE = 500  # Semantic Scholar batch endpoint limit


def fetch_citation_counts(pmids: list[str]) -> dict[str, int]:
    """
    Fetch citation counts for a list of PMIDs via Semantic Scholar batch API.
    Returns {pmid: citation_count}. PMIDs with no S2 record are omitted.
    """
    if not pmids:
        return {}

    result: dict[str, int] = {}

    for i in range(0, len(pmids), _BATCH_SIZE):
        batch = pmids[i : i + _BATCH_SIZE]
        ids = [f"PMID:{pmid}" for pmid in batch]

        for attempt in range(3):
            try:
                resp = httpx.post(
                    f"{SEMANTIC_SCHOLAR_BASE}/paper/batch",
                    params={"fields": "citationCount,externalIds"},
                    json={"ids": ids},
                    timeout=30,
                )
                resp.raise_for_status()
                papers = resp.json()
                break
            except httpx.HTTPError as exc:
                if attempt == 2:
                    _logger.warning(f"Semantic Scholar batch failed: {exc}")
                    papers = []
                    break
                time.sleep(2 ** attempt)

        for paper in papers:
            if paper is None:
                continue
            ext = paper.get("externalIds") or {}
            pmid = ext.get("PubMed")
            count = paper.get("citationCount")
            if pmid and count is not None:
                result[str(pmid)] = int(count)

        # Free tier allows ~100 requests per 5 minutes; 1s delay keeps us safe
        time.sleep(1.0)

    _logger.info(
        "Semantic Scholar citation fetch",
        extra={"data": {"requested": len(pmids), "found": len(result)}},
    )
    return result
