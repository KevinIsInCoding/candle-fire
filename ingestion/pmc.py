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
    time.sleep(0.1 if os.getenv("NCBI_API_KEY") else 0.4)


def get_pmcids(pmids: list[str]) -> dict[str, str]:
    """
    Map PubMed IDs to PMC IDs for papers with Open Access full text.
    Returns {pmid: pmcid}.
    """
    _configure_entrez()
    if not pmids:
        return {}

    result: dict[str, str] = {}
    for i in range(0, len(pmids), 200):
        batch = pmids[i : i + 200]
        for attempt in range(3):
            try:
                handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=",".join(batch))
                link_sets = Entrez.read(handle)
                handle.close()
                break
            except Exception as exc:
                if attempt == 2:
                    _logger.warning(f"elink failed: {exc}")
                    link_sets = []
                    break
                time.sleep(2 ** attempt)

        for link_set in link_sets:
            source_ids = link_set.get("IdList", [])
            source_id = str(source_ids[0]) if source_ids else None
            for db_link in link_set.get("LinkSetDb", []):
                if db_link.get("DbTo") == "pmc":
                    links = db_link.get("Link", [])
                    if links and source_id:
                        result[source_id] = str(links[0]["Id"])
                    break
        _sleep()

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
    for sec in body.findall(".//sec"):
        # Skip nested sections — only top-level sec elements under body
        if sec in body:
            title_el = sec.find("title")
            title = (title_el.text or "Section").strip() if title_el is not None else "Section"

            paragraphs: list[str] = []
            for p in sec.findall("p"):
                text = "".join(p.itertext()).strip()
                if text:
                    paragraphs.append(text)

            if paragraphs:
                sections.append(f"[{title}]\n" + "\n".join(paragraphs))

    return "\n\n".join(sections) if sections else None
