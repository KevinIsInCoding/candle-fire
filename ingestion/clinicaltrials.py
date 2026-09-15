"""ClinicalTrials.gov v2 client for ALS trials (adapted from beacon/trials_api.py)."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

import httpx

from config import CTGOV_BASE, EXTRACTION_MODEL
from logging_config import get_logger
from prompts import TRIAL_EXTRACTION_SYSTEM, TRIAL_SUMMARY_SYSTEM

if TYPE_CHECKING:
    import anthropic

_logger = get_logger("ingestion.clinicaltrials")

_TRIAL_BATCH_SIZE = 10


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


def _extract_eligibility(elig_mod: dict) -> dict:
    """Enrollment/eligibility fields from a CT.gov v2 eligibilityModule.

    Shared by _flatten_trial (ingestion) and scripts/backfill_eligibility.py so the stored
    shape is identical whichever path populated it.
    """
    return {
        "criteria": elig_mod.get("eligibilityCriteria", ""),  # free text: Inclusion/Exclusion
        "sex": elig_mod.get("sex", ""),                        # ALL / MALE / FEMALE
        "min_age": elig_mod.get("minimumAge", ""),             # e.g. "18 Years"
        "max_age": elig_mod.get("maximumAge", ""),
        "healthy_volunteers": elig_mod.get("healthyVolunteers"),
        "std_ages": elig_mod.get("stdAges", []),               # e.g. ["ADULT", "OLDER_ADULT"]
    }


def _flatten_trial(study: dict) -> dict:
    proto = study.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    desc_mod = proto.get("descriptionModule", {})
    design_mod = proto.get("designModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    arms_mod = proto.get("armsInterventionsModule", {})
    status_mod = proto.get("statusModule", {})
    contacts_mod = proto.get("contactsLocationsModule", {})
    elig_mod = proto.get("eligibilityModule", {})

    nct_id = id_mod.get("nctId", "")
    interventions = [
        {"type": iv.get("type", ""), "name": iv.get("name", "")}
        for iv in arms_mod.get("interventions", [])
    ]

    study_type = design_mod.get("studyType", "")
    is_eap = study_type == "EXPANDED_ACCESS"

    # Site locations — facility, address, per-site recruiting status, and geo point.
    # Physicians search trials by facility ("Mass General") or place ("in NY"), so this
    # address data must be persisted in the trials DB (offline; no fetch at query time).
    locations = [
        {
            "facility": (loc.get("facility") or "").strip(),
            "city": loc.get("city", ""),
            "state": loc.get("state", ""),
            "country": loc.get("country", ""),
            "status": loc.get("status", ""),  # per-site recruiting status
            "lat": (loc.get("geoPoint") or {}).get("lat"),
            "lon": (loc.get("geoPoint") or {}).get("lon"),
        }
        for loc in contacts_mod.get("locations", [])
    ]

    central_contacts = contacts_mod.get("centralContacts", [])
    contact_phone = next((c.get("phone", "") for c in central_contacts if c.get("phone")), "")
    contact_email = next((c.get("email", "") for c in central_contacts if c.get("email")), "")

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
        "locations": locations,
        "contact_phone": contact_phone,
        "contact_email": contact_email,
        "eligibility": _extract_eligibility(elig_mod),
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


_SUMMARY_BATCH_SIZE = 5          # trials per Claude call (each carries its own evidence block)
_SUMMARY_EVIDENCE_PER_TRIAL = 8  # top retrieved passages offered to the model per trial
_SUMMARY_PASSAGE_CHARS = 600     # per-passage text budget in the prompt
_SUMMARY_CLAIM_FIELDS = ("targeting_mechanism", "animal_results", "repurposed_from")


def enrich_trial_mechanisms(
    trials: list[dict],
    collection,
    client: "anthropic.Anthropic",
    checkpoint=None,
) -> None:
    """Attach a RAG-grounded `mechanism_summary` to each trial; mutates in-place.

    Runs as offline pipeline step 5.5 (after build_index) because it grounds animal-results and
    repurposed-from claims in the ChromaDB corpus — citing a passage's PMID or falling back to
    "unknown". Resumable: trials that already carry a `mechanism_summary` are skipped, and
    `checkpoint(trials)` (if given) is called after each batch so an interrupted run keeps its work.
    """
    from rich.progress import (
        BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn,
    )

    pending = [t for t in trials if not t.get("mechanism_summary")]
    if not pending:
        _logger.info("all trials already have a mechanism_summary; nothing to do")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Summarizing trial mechanisms (RAG)...", total=len(pending))

        for i in range(0, len(pending), _SUMMARY_BATCH_SIZE):
            batch = pending[i : i + _SUMMARY_BATCH_SIZE]
            evidence = {t["nct_id"]: _retrieve_trial_evidence(t, collection) for t in batch}
            results = _call_summary_batch(client, batch, evidence)

            for trial in batch:
                raw = results.get(trial["nct_id"])
                trial["mechanism_summary"] = _sanitize_summary(raw, evidence[trial["nct_id"]], trial)

            progress.advance(task, len(batch))
            if checkpoint is not None:
                checkpoint(trials)
            if i + _SUMMARY_BATCH_SIZE < len(pending):
                time.sleep(1.0)


def _retrieve_trial_evidence(trial: dict, collection) -> list[dict]:
    """Top corpus passages for a trial's compound/target — the grounding pool for its summary."""
    if collection is None or collection.count() == 0:
        return []
    from rag import retriever as rag_retriever

    entities = list(trial.get("target_entities") or [])
    iv_names = [iv.get("name", "") for iv in trial.get("interventions", []) if iv.get("name")]

    results: list[dict] = []
    if entities:
        results = rag_retriever.search_by_entities(collection, entities)
    # Compound-name keyword search catches drug-specific papers embeddings miss (drug codes).
    if iv_names:
        seen = {r["pmid"] for r in results}
        for r in rag_retriever.search_by_keyword(collection, iv_names):
            if r["pmid"] not in seen:
                results.append(r)
                seen.add(r["pmid"])
    # Preclinical-focused query so animal-model abstracts surface — without it, the
    # target/compound passages seldom state animal results and the field stays "unknown".
    if iv_names:
        seen = {r["pmid"] for r in results}
        precl_q = f"{iv_names[0]} mouse model preclinical survival motor neuron ALS"
        for r in rag_retriever.search(collection, precl_q):
            if r["pmid"] not in seen:
                results.append(r)
                seen.add(r["pmid"])
    if not results:
        query = f"{trial.get('title', '')} {' '.join(iv_names)}".strip()
        results = rag_retriever.search(collection, query) if query else []

    results = rag_retriever.apply_citation_boost(results)
    return results[:_SUMMARY_EVIDENCE_PER_TRIAL]


def _sanitize_summary(raw: dict | None, evidence: list[dict], trial: dict) -> dict:
    """Coerce the model output into the stored shape and enforce the grounding guardrail.

    Every claim field defaults to "unknown"; any cited PMID that is not in this trial's evidence
    pool is dropped, and animal_results / repurposed_from are downgraded to "unknown" when they
    lose their citation — so a hallucinated or uncited claim can never reach a physician.
    """
    allowed = {str(r.get("pmid", "")) for r in evidence if r.get("pmid")}
    fallback_compound = next(
        (iv.get("name", "") for iv in trial.get("interventions", []) if iv.get("name")), ""
    )
    out = {
        "compound": (raw or {}).get("compound") or fallback_compound or "unknown",
        "targeting_mechanism": "unknown",
        "targeting_mechanism_pmid": "",
        "animal_results": "unknown",
        "animal_results_pmid": "",
        "repurposed_from": "unknown",
        "repurposed_from_pmid": "",
    }
    if not raw:
        return out

    for field_name in _SUMMARY_CLAIM_FIELDS:
        value = (raw.get(field_name) or "").strip()
        pmid = str(raw.get(f"{field_name}_pmid") or "").strip()
        if pmid and pmid not in allowed:
            pmid = ""  # cited a paper we never showed it — drop the citation
        # animal_results / repurposed_from are corpus-only: no valid citation ⇒ not trustworthy.
        if field_name != "targeting_mechanism" and value.lower() not in ("", "unknown", "not repurposed") and not pmid:
            value = "unknown"
        out[field_name] = value or "unknown"
        out[f"{field_name}_pmid"] = pmid
    return out


def _call_summary_batch(
    client: "anthropic.Anthropic",
    batch: list[dict],
    evidence: dict[str, list[dict]],
) -> dict[str, dict]:
    """Send one batch of trials + their evidence to Claude; return {nct_id: summary input dict}."""
    from tools import TRIAL_SUMMARY_TOOLS

    lines = [
        f"Summarize the mechanism for each of these {len(batch)} ALS trials. "
        "Call summarize_trial_mechanism once per trial.\n"
    ]
    for trial in batch:
        nct = trial["nct_id"]
        iv_names = ", ".join(iv["name"] for iv in trial.get("interventions", [])) or "N/A"
        summary = (trial.get("summary") or "")[:500]
        ev_lines = [
            f"  [PMID {r.get('pmid', '')}] {r.get('title', '')} ({r.get('year') or 'n.d.'}): "
            f"{(r.get('document') or '')[:_SUMMARY_PASSAGE_CHARS]}"
            for r in evidence.get(nct, [])
        ] or ["  (no passages retrieved — animal_results and repurposed_from must be 'unknown')"]
        lines.append(
            f"--- NCT: {nct} ---\n"
            f"Title: {trial['title']}\n"
            f"Interventions: {iv_names}\n"
            f"Summary: {summary}\n"
            f"EVIDENCE:\n" + "\n".join(ev_lines) + "\n"
        )

    for attempt in range(3):
        try:
            response = client.messages.create(
                model=EXTRACTION_MODEL,
                max_tokens=4096,
                system=TRIAL_SUMMARY_SYSTEM,
                tools=TRIAL_SUMMARY_TOOLS,
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": "\n".join(lines)}],
            )
            break
        except Exception as exc:
            if attempt == 2:
                _logger.warning(f"Claude trial-summary failed: {exc}")
                return {}
            time.sleep(30 if "rate" in str(exc).lower() else 2 ** attempt)

    results: dict[str, dict] = {}
    for block in response.content:
        if block.type == "tool_use" and block.name == "summarize_trial_mechanism":
            nct_id = block.input.get("nct_id", "")
            if nct_id:
                results[nct_id] = block.input
    return results


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
                system=TRIAL_EXTRACTION_SYSTEM,
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
