"""ClinicalTrials.gov v2 client for ALS trials (adapted from beacon/trials_api.py)."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import httpx

from config import CTGOV_BASE, EXTRACTION_MODEL
from logging_config import get_logger

if TYPE_CHECKING:
    import anthropic

_logger = get_logger("ingestion.clinicaltrials")

_TRIAL_BATCH_SIZE = 10

_TRIAL_EXTRACTION_SYSTEM = """You are a biomedical NLP expert specializing in ALS (amyotrophic lateral sclerosis) clinical trials.

For each trial provided, identify the primary biological target(s) being tested or modulated:
- Genes silenced or corrected (e.g., SOD1, TARDBP, FUS, C9orf72, NEK1, VCP, TBK1)
- Proteins targeted (use canonical gene symbol, e.g. TARDBP for TDP-43 protein)
- Compounds/drugs — report the molecular or pathway target, not the drug name (e.g., a trial of AMX0114 targets TARDBP)
- Mechanisms (e.g., neuroinflammation, oxidative stress, glutamate excitotoxicity)

Call extract_trial_targets once per trial. Return an empty targets list only when no specific molecular or mechanistic target is identifiable."""


def fetch_als_trials(
    client: "anthropic.Anthropic | None" = None,
) -> list[dict]:
    """Fetch all ALS interventional + expanded-access studies, regardless of status.

    No status filter — completed, terminated, and withdrawn trials are as
    clinically important as active ones (negative results inform research).
    `studyType:int exp` includes both interventional trials AND Expanded Access
    Programs (EAP / compassionate use, e.g. NCT05281484), which physicians need for
    off-trial access options. Status filtering is left to query time.
    """
    params: dict[str, str | int] = {
        "query.cond": "Amyotrophic Lateral Sclerosis",
        "aggFilters": "studyType:int exp",
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
                _logger.warning(f"ClinicalTrials.gov error (attempt {attempt + 1}): {exc}")
                time.sleep(2 ** attempt)

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

    if client is not None:
        _enrich_targets_llm(trials, client)

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
    interventions = [
        {"type": iv.get("type", ""), "name": iv.get("name", "")}
        for iv in arms_mod.get("interventions", [])
    ]

    study_type = design_mod.get("studyType", "")
    is_eap = study_type == "EXPANDED_ACCESS"

    return {
        "nct_id": nct_id,
        "title": id_mod.get("briefTitle", ""),
        # Expanded Access has no trial phase — surface it as the phase label instead
        "phase": "Expanded Access" if is_eap else (", ".join(design_mod.get("phases", [])) or "N/A"),
        "status": status_mod.get("overallStatus", ""),
        "study_type": study_type,
        "is_expanded_access": is_eap,
        "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
        "summary": desc_mod.get("briefSummary", ""),
        "interventions": interventions,
        "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
        "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
        "target_entities": [],
    }


def _enrich_targets_llm(trials: list[dict], client: "anthropic.Anthropic") -> None:
    """Call Claude in batches to extract biological targets; mutates each trial in-place."""
    from extraction.normalizer import normalize_entity
    from tools import TRIAL_EXTRACTION_TOOLS

    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Extracting trial targets (LLM)...", total=len(trials))

        for i in range(0, len(trials), _TRIAL_BATCH_SIZE):
            batch = trials[i : i + _TRIAL_BATCH_SIZE]
            results = _call_claude_batch(client, batch, TRIAL_EXTRACTION_TOOLS)

            for nct_id, raw_targets in results.items():
                trial = next((t for t in batch if t["nct_id"] == nct_id), None)
                if trial is None:
                    continue
                canonical: list[str] = []
                for t in raw_targets:
                    if t.get("confidence", 0) < 0.5:
                        continue
                    canon_id = normalize_entity(t["name"], t["type"])
                    # strip prefix (e.g. "protein:TARDBP" → "TARDBP")
                    canon_name = canon_id.split(":", 1)[-1]
                    if canon_name and canon_name not in canonical:
                        canonical.append(canon_name)
                trial["target_entities"] = canonical

            progress.advance(task, len(batch))

            if i + _TRIAL_BATCH_SIZE < len(trials):
                time.sleep(1.0)


def _call_claude_batch(
    client: "anthropic.Anthropic",
    batch: list[dict],
    tools: list,
) -> dict[str, list[dict]]:
    """Send one batch of trials to Claude; return {nct_id: [target dicts]}."""
    lines = [
        f"Extract targets from each of the following {len(batch)} ALS clinical trials. "
        "Call extract_trial_targets once per trial.\n"
    ]
    for trial in batch:
        iv_names = ", ".join(iv["name"] for iv in trial.get("interventions", [])) or "N/A"
        summary = (trial.get("summary") or "")[:400]
        lines.append(
            f"--- NCT: {trial['nct_id']} ---\n"
            f"Title: {trial['title']}\n"
            f"Interventions: {iv_names}\n"
            f"Summary: {summary}\n"
        )

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=EXTRACTION_MODEL,
                max_tokens=4096,
                system=_TRIAL_EXTRACTION_SYSTEM,
                tools=tools,
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": "\n".join(lines)}],
            )
            break
        except Exception as exc:
            if attempt == 2:
                _logger.warning(f"Claude trial extraction failed: {exc}")
                return {}
            time.sleep(30 if "rate" in str(exc).lower() else 2 ** attempt)

    results: dict[str, list[dict]] = {}
    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_trial_targets":
            nct_id = block.input.get("nct_id", "")
            if nct_id:
                results[nct_id] = block.input.get("targets", [])

    return results
