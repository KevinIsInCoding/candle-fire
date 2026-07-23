"""PubMed Entrez API client for ALS paper ingestion."""
from __future__ import annotations

import os
import time

from Bio import Entrez, Medline

from config import PUBMED_BATCH_SIZE
from logging_config import get_logger
from models import ALSPaper

_logger = get_logger("ingestion.pubmed")


def _configure_entrez() -> None:
    email = os.environ.get("ENTREZ_EMAIL")
    if not email:
        raise EnvironmentError("ENTREZ_EMAIL environment variable is required by NCBI")
    Entrez.email = email
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key


def _sleep() -> None:
    """Respect NCBI rate limits: 10 req/s with API key, 3 req/s without."""
    time.sleep(0.1 if os.getenv("NCBI_API_KEY") else 0.4)


_ESEARCH_PAGE_SIZE = 9999  # NCBI hard cap per esearch call


def search_pmids(query: str, max_results: int = 500) -> list[str]:
    """Search PubMed with a query string and return a list of PMIDs.

    Pages through esearch results in chunks of 9,999 (NCBI's per-call cap)
    until max_results or the total result count is reached.
    """
    _configure_entrez()
    pmids: list[str] = []
    retstart = 0
    total: int | None = None

    while True:
        want = min(_ESEARCH_PAGE_SIZE, max_results - len(pmids))
        handle = Entrez.esearch(db="pubmed", term=query, retmax=want, retstart=retstart)
        record = Entrez.read(handle)
        handle.close()

        if total is None:
            total = int(record["Count"])

        page = list(record["IdList"])
        pmids.extend(page)

        if not page or len(pmids) >= max_results or len(pmids) >= total:
            break

        retstart += len(page)
        _sleep()

    _logger.info("PubMed esearch", extra={"data": {"count": len(pmids), "total": total, "query": query[:80]}})
    return pmids


def fetch_by_pmids(pmids: list[str]) -> list[ALSPaper]:
    """Fetch and parse paper records for a list of PMIDs."""
    _configure_entrez()
    papers: list[ALSPaper] = []

    for i in range(0, len(pmids), PUBMED_BATCH_SIZE):
        batch = pmids[i : i + PUBMED_BATCH_SIZE]
        _logger.debug("Fetching Entrez batch", extra={"data": {"batch": i // PUBMED_BATCH_SIZE + 1, "size": len(batch)}})

        for attempt in range(3):
            try:
                handle = Entrez.efetch(db="pubmed", id=",".join(batch), rettype="medline", retmode="text")
                records = list(Medline.parse(handle))
                handle.close()
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                _logger.warning(f"Entrez fetch error (attempt {attempt + 1}): {exc}")
                time.sleep(wait)

        for record in records:
            paper = _parse_record(record)
            if paper:
                papers.append(paper)
        _sleep()

    _logger.info("PubMed fetch complete", extra={"data": {"total": len(papers)}})
    return papers


def _parse_record(record: dict) -> ALSPaper | None:
    """Convert a Biopython Medline record to ALSPaper. Returns None if no abstract."""
    pmid = record.get("PMID", "")
    abstract = record.get("AB", "")
    if not pmid or not abstract:
        return None

    authors = record.get("FAU", record.get("AU", []))
    if isinstance(authors, str):
        authors = [authors]

    # "DP" field: "2023 Jan 15", "2023 Jan", "2023"
    year = 0
    date_str = record.get("DP", "")
    if date_str:
        try:
            year = int(date_str.split()[0])
        except (ValueError, IndexError):
            pass

    # DOI from AID list: ["10.1093/xxx [doi]", "S0092-8674(23)00001-1 [pii]"]
    doi = ""
    for aid in record.get("AID", []):
        if aid.endswith("[doi]"):
            doi = aid.replace(" [doi]", "").strip()
            break

    return ALSPaper(
        pmid=pmid,
        title=record.get("TI", ""),
        abstract=abstract,
        authors=authors,
        year=year,
        doi=doi,
        mesh_terms=record.get("MH", []),
    )
