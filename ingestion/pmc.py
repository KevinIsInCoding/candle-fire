"""PMC XML full-text fetcher for Open Access papers."""
from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET

from Bio import Entrez

from logging_config import get_logger

_logger = get_logger("ingestion.pmc")


def _configure_entrez() -> None:
    email = os.environ.get("ENTREZ_EMAIL")
    if not email:
        raise EnvironmentError("ENTREZ_EMAIL environment variable is required")
    Entrez.email = email
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key


def _sleep() -> None:
    # NCBI rate limit: 10 req/s with API key, 3 req/s without
    time.sleep(0.1 if os.getenv("NCBI_API_KEY") else 0.4)


_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
_IDCONV_BATCH = 200


def get_pmcids(pmids: list[str]) -> dict[str, str]:
    """
    Map PubMed IDs to PMC IDs for papers with Open Access full text.
    Uses the NCBI ID Converter API (idconv) which accepts batches of 200 PMIDs
    and returns a proper PMID→PMCID mapping — dramatically faster than one
    elink call per PMID.
    Returns {pmid: pmcid}.
    """
    import urllib.request
    import urllib.parse

    if not pmids:
        return {}

    from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, TextColumn

    email = os.environ.get("ENTREZ_EMAIL", "")
    result: dict[str, str] = {}
    batches = [pmids[i : i + _IDCONV_BATCH] for i in range(0, len(pmids), _IDCONV_BATCH)]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Looking up PMC IDs...", total=len(batches))

        for batch in batches:
            params = urllib.parse.urlencode({
                "ids": ",".join(batch),
                "format": "json",
                "email": email,
            })
            url = f"{_IDCONV_URL}?{params}"
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=30) as resp:
                        import json as _json
                        data = _json.loads(resp.read())
                    break
                except Exception as exc:
                    if attempt == 2:
                        _logger.warning(f"idconv batch failed: {exc}")
                        data = {}
                        break
                    time.sleep(2 ** attempt)

            for record in data.get("records", []):
                pmid = record.get("pmid")
                pmcid = record.get("pmcid")
                # pmid comes back as int from the API; pmcid is like "PMC1234567"
                if pmid and pmcid and pmcid.startswith("PMC"):
                    result[str(pmid)] = pmcid[3:]

            time.sleep(0.4)
            progress.advance(task)

    _logger.info("PMC ID lookup", extra={"data": {"pmids": len(pmids), "found": len(result)}})
    return result


def fetch_full_text(pmcid: str) -> str | None:
    """
    Fetch PMC XML for a single PMCID and parse into section-labeled text.
    Returns None if the fetch fails or the article has no body sections.
    """
    _configure_entrez()
    for attempt in range(3):
        try:
            handle = Entrez.efetch(db="pmc", id=pmcid, rettype="xml", retmode="xml")
            xml_data = handle.read()
            handle.close()
            break
        except Exception as exc:
            if attempt == 2:
                _logger.warning(f"PMC fetch failed for PMCID {pmcid}: {exc}")
                return None
            time.sleep(2 ** attempt)
    _sleep()
    return _parse_jats_xml(xml_data)


def _parse_jats_xml(xml_data: bytes) -> str | None:
    """Extract section-labeled text from JATS/PMC XML."""
    try:
        root = ET.fromstring(xml_data)
    except ET.ParseError:
        return None

    body = root.find(".//body")
    if body is None:
        return None

    sections: list[str] = []

    # Try structured sections first (standard JATS)
    top_secs = [sec for sec in body.findall(".//sec") if sec in body]
    for sec in top_secs:
        title_el = sec.find("title")
        title = (title_el.text or "Section").strip() if title_el is not None else "Section"
        paragraphs = ["".join(p.itertext()).strip() for p in sec.findall(".//p")]
        paragraphs = [t for t in paragraphs if t]
        if paragraphs:
            sections.append(f"[{title}]\n" + "\n".join(paragraphs))

    # Fall back to bare paragraphs when there are no top-level sec elements
    if not sections:
        paragraphs = ["".join(p.itertext()).strip() for p in body.findall(".//p")]
        paragraphs = [t for t in paragraphs if t]
        if paragraphs:
            sections.append("[Body]\n" + "\n".join(paragraphs))

    return "\n\n".join(sections) if sections else None
