"""ClinicalTrials.gov v2 client for ALS trials (adapted from beacon/trials_api.py)."""
from __future__ import annotations

import re
import time

import httpx

from config import CTGOV_BASE
from logging_config import get_logger

_logger = get_logger("ingestion.clinicaltrials")

# Known ALS-relevant targets for heuristic entity linking
_KNOWN_TARGETS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bSOD1\b", re.IGNORECASE), "SOD1"),
    (re.compile(r"\bTARDBP\b", re.IGNORECASE), "TARDBP"),
    (re.compile(r"\bTDP-?43\b", re.IGNORECASE), "TARDBP"),
    (re.compile(r"\bFUS\b", re.IGNORECASE), "FUS"),
    (re.compile(r"\bC9orf72\b", re.IGNORECASE), "C9orf72"),
    (re.compile(r"\bATXN2\b", re.IGNORECASE), "ATXN2"),
    (re.compile(r"\bTBK1\b", re.IGNORECASE), "TBK1"),
    (re.compile(r"\bNEK1\b", re.IGNORECASE), "NEK1"),
    (re.compile(r"\bVCP\b", re.IGNORECASE), "VCP"),
    (re.compile(r"\briluzole\b", re.IGNORECASE), "riluzole"),
    (re.compile(r"\bedaravone\b", re.IGNORECASE), "edaravone"),
    (re.compile(r"\btofersen\b", re.IGNORECASE), "tofersen"),
    (re.compile(r"\bAMX0035\b", re.IGNORECASE), "AMX0035"),
    (re.compile(r"\bmasitinib\b", re.IGNORECASE), "masitinib"),
    (re.compile(r"\bbosutinib\b", re.IGNORECASE), "bosutinib"),
    (re.compile(r"\bmexiletine\b", re.IGNORECASE), "mexiletine"),
    (re.compile(r"\bantisense oligonucleotide\b", re.IGNORECASE), "antisense oligonucleotide"),
    (re.compile(r"\bASO\b"), "antisense oligonucleotide"),
    (re.compile(r"\bsiRNA\b", re.IGNORECASE), "siRNA"),
    (re.compile(r"\bstem cell\b", re.IGNORECASE), "stem cell"),
    (re.compile(r"\bgene therapy\b", re.IGNORECASE), "gene therapy"),
]


def fetch_als_trials(status: str = "RECRUITING") -> list[dict]:
    """Fetch ALS interventional trials. Returns flat dicts ready for JSONL serialization."""
    params: dict[str, str | int] = {
        "query.cond": "Amyotrophic Lateral Sclerosis",
        "filter.overallStatus": status,
        "aggFilters": "studyType:int",
        "pageSize": 1000,
        "format": "json",
    }

    all_studies: list[dict] = []
    while True:
        for attempt in range(3):
            try:
                resp = httpx.get(CTGOV_BASE, params=params, timeout=30)
                resp.raise_for_status()
                body = resp.json()
                break
            except httpx.HTTPError as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                _logger.warning(f"ClinicalTrials.gov error (attempt {attempt + 1}): {exc}")
                time.sleep(wait)

        page_studies = body.get("studies", [])
        all_studies.extend(page_studies)
        next_token = body.get("nextPageToken")
        _logger.debug(
            "ClinicalTrials.gov page",
            extra={"data": {"count": len(page_studies), "has_next": bool(next_token)}},
        )
        if not next_token:
            break
        params["pageToken"] = next_token

    trials = [_flatten_trial(s) for s in all_studies]
    _logger.info("ALS trial fetch complete", extra={"data": {"total": len(trials)}})
    return trials


def _flatten_trial(study: dict) -> dict:
    proto = study.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    desc_mod = proto.get("descriptionModule", {})
    design_mod = proto.get("designModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    arms_mod = proto.get("armsInterventionsModule", {})
    status_mod = proto.get("statusModule", {})

    nct_id = id_mod.get("nctId", "")
    title = id_mod.get("briefTitle", "")
    interventions = [
        {"type": iv.get("type", ""), "name": iv.get("name", "")}
        for iv in arms_mod.get("interventions", [])
    ]
    intervention_names = " ".join(iv["name"] for iv in interventions)

    return {
        "nct_id": nct_id,
        "title": title,
        "phase": ", ".join(design_mod.get("phases", [])) or "N/A",
        "status": status_mod.get("overallStatus", ""),
        "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
        "summary": desc_mod.get("briefSummary", ""),
        "interventions": interventions,
        "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
        "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
        "target_entities": extract_target_entities(f"{title} {intervention_names}"),
    }


def extract_target_entities(text: str) -> list[str]:
    """Scan text for known ALS target names. Returns sorted canonical entity names."""
    found: set[str] = set()
    for pattern, canonical in _KNOWN_TARGETS:
        if pattern.search(text):
            found.add(canonical)
    return sorted(found)
