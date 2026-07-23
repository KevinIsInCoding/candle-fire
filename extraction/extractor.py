"""
Claude Haiku entity extractor — Batch API.

Submits extraction requests through the Message Batches API (50% cheaper than
synchronous calls) since this is an offline, non-latency-sensitive pipeline.
Batches 20 papers per request; uses full_text when available, else abstract.
Resumable at two levels: completed PMIDs are tracked in .progress.json, and an
in-flight batch id is persisted in .batch_state.json so an interrupted run
resumes polling the same (already-paid-for) batch instead of resubmitting.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from config import (
    ENTITIES_PATH,
    EXTRACTION_BATCH_SIZE,
    EXTRACTION_BATCH_STATE_PATH,
    EXTRACTION_MODEL,
    EXTRACTION_PROGRESS_PATH,
    PAPERS_PATH,
)
from extraction.normalizer import CanonicalRegistry, guess_entity_type, normalize_entity
from logging_config import get_logger
from models import ALSPaper, ExtractedEntity, EntityRelationship, PaperExtractionResult
from tools import EXTRACTION_TOOLS

_logger = get_logger("extraction.extractor")

# Seconds between batch status polls. Batches usually finish in well under an
# hour; the ceiling is 24h.
_POLL_INTERVAL_S = 30

_EXTRACTION_SYSTEM = """\
You are a biomedical NLP expert specializing in ALS (amyotrophic lateral sclerosis).
Extract entities and relationships from each paper using the extract_entities tool.
Call it once per paper. Use the full text when provided — it is richer than the abstract alone.

Entity types: Gene, Protein, Compound, Pathway, Phenotype, Mechanism.
Relationship types: BINDS, INHIBITS, ASSOCIATED_WITH, TESTED_IN, EXPRESSED_IN, CO_OCCURS.

Be precise. Only extract entities explicitly mentioned. Return pmid exactly as given.
"""


def extract_all(
    papers_path: Path = PAPERS_PATH,
    entities_path: Path = ENTITIES_PATH,
    progress_path: Path = EXTRACTION_PROGRESS_PATH,
    batch_state_path: Path = EXTRACTION_BATCH_STATE_PATH,
    client: anthropic.Anthropic | None = None,
) -> list[PaperExtractionResult]:
    """Extract entities from all papers via the Batch API. Skips done PMIDs.

    Runs one main batch round (20 papers/request), then an individual retry
    round for any papers Claude skipped, then records empty results for papers
    still missing so they aren't re-attempted on the next run.
    """
    if client is None:
        client = anthropic.Anthropic()

    papers = _load_papers(papers_path)
    paper_by_pmid = {p.pmid: p for p in papers}
    done_pmids = _load_progress(progress_path)

    pending = [p for p in papers if p.pmid not in done_pmids]
    _logger.info(
        f"{len(papers)} papers total; {len(done_pmids)} already processed; {len(pending)} pending"
    )

    if not pending:
        return []

    registry = CanonicalRegistry()
    entities_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[PaperExtractionResult] = []
    with open(entities_path, "a", encoding="utf-8") as out_f:
        # Round 1 — main batches of EXTRACTION_BATCH_SIZE papers each.
        batches = [
            pending[i : i + EXTRACTION_BATCH_SIZE]
            for i in range(0, len(pending), EXTRACTION_BATCH_SIZE)
        ]
        main_map = {f"batch-{i}": batch for i, batch in enumerate(batches)}
        round1 = _run_batch_round(client, main_map, registry, paper_by_pmid, batch_state_path)
        _write_results(out_f, round1, done_pmids, progress_path, registry)
        all_results.extend(round1)

        found = {r.pmid for r in round1}
        missed = [p for p in pending if p.pmid not in found]

        # Round 2 — retry missed papers one per request.
        if missed:
            _logger.info(f"Retrying {len(missed)} missed papers individually")
            retry_map = {f"retry-{p.pmid}": [p] for p in missed}
            round2 = _run_batch_round(client, retry_map, registry, paper_by_pmid, batch_state_path)
            _write_results(out_f, round2, done_pmids, progress_path, registry)
            all_results.extend(round2)
            found |= {r.pmid for r in round2}

        # Record empty results for anything still missing after retry.
        still_missing = [p for p in pending if p.pmid not in found]
        if still_missing:
            empties = []
            for p in still_missing:
                _logger.warning(
                    f"No extraction result for PMID {p.pmid} after retry — recording empty"
                )
                empties.append(PaperExtractionResult(pmid=p.pmid, entities=[], relationships=[]))
            _write_results(out_f, empties, done_pmids, progress_path, registry)
            all_results.extend(empties)

    return all_results


def _run_batch_round(
    client: anthropic.Anthropic,
    custom_id_to_papers: dict[str, list[ALSPaper]],
    registry: CanonicalRegistry,
    paper_by_pmid: dict[str, ALSPaper],
    state_path: Path,
) -> list[PaperExtractionResult]:
    """Submit (or resume) one batch, poll to completion, and parse its results.

    Persists the batch id + custom_id→PMID mapping to state_path on submit so an
    interrupted process resumes the same batch. Clears the state on completion.
    """
    batch = None
    state = _load_batch_state(state_path)
    if state and state.get("batch_id"):
        try:
            existing = client.messages.batches.retrieve(state["batch_id"])
        except anthropic.NotFoundError:
            _logger.warning("Persisted batch id not found — submitting a fresh batch")
        else:
            if existing.processing_status in {"in_progress", "validating", "finalizing", "ended"}:
                _logger.info(f"Resuming in-flight batch {existing.id}")
                batch = existing
                # Rebuild the mapping from persisted PMIDs so results match.
                custom_id_to_papers = {
                    cid: [paper_by_pmid[pmid] for pmid in pmids if pmid in paper_by_pmid]
                    for cid, pmids in state.get("papers", {}).items()
                }

    if batch is None:
        requests = [
            Request(custom_id=cid, params=_build_params(papers))
            for cid, papers in custom_id_to_papers.items()
        ]
        batch = client.messages.batches.create(requests=requests)
        _save_batch_state(
            state_path,
            {
                "batch_id": batch.id,
                "papers": {
                    cid: [p.pmid for p in papers] for cid, papers in custom_id_to_papers.items()
                },
            },
        )
        _logger.info(f"Submitted batch {batch.id} with {len(requests)} requests")

    batch = _poll_until_done(client, batch)

    results: list[PaperExtractionResult] = []
    for res in client.messages.batches.results(batch.id):
        papers = custom_id_to_papers.get(res.custom_id, [])
        local_by_pmid = {p.pmid: p for p in papers}
        if res.result.type == "succeeded":
            results.extend(
                _parse_response_blocks(res.result.message.content, local_by_pmid, registry)
            )
        elif res.result.type == "errored":
            _logger.warning(f"Batch request {res.custom_id} errored: {res.result.error}")
        else:
            _logger.warning(f"Batch request {res.custom_id} {res.result.type}")

    _clear_batch_state(state_path)
    return results


def _poll_until_done(client: anthropic.Anthropic, batch) -> object:
    """Poll a batch until it reaches a terminal status, showing progress."""
    total = (
        batch.request_counts.processing
        + batch.request_counts.succeeded
        + batch.request_counts.errored
        + batch.request_counts.canceled
        + batch.request_counts.expired
    )
    with Progress(
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Extracting entities (batch)", total=total or None)
        while batch.processing_status != "ended":
            if batch.processing_status in {"canceling", "canceled", "expired"}:
                _logger.warning(f"Batch {batch.id} ended early with status {batch.processing_status}")
                break
            time.sleep(_POLL_INTERVAL_S)
            batch = client.messages.batches.retrieve(batch.id)
            counts = batch.request_counts
            completed = counts.succeeded + counts.errored + counts.canceled + counts.expired
            progress.update(task, completed=completed)
        progress.update(task, completed=total)
    return batch


def _build_params(batch: list[ALSPaper]) -> MessageCreateParamsNonStreaming:
    """Build the per-request Messages params for a batch of papers.

    system + tools are identical across every request, but on Haiku 4.5 the
    combined prefix is far below the 4096-token minimum cacheable size, so
    prompt caching would silently no-op — we don't set cache_control here.
    """
    return MessageCreateParamsNonStreaming(
        model=EXTRACTION_MODEL,
        max_tokens=8192,
        system=_EXTRACTION_SYSTEM,
        tools=list(EXTRACTION_TOOLS),
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": _format_batch(batch)}],
    )


def _parse_response_blocks(
    blocks: list,
    paper_by_pmid: dict[str, ALSPaper],
    registry: CanonicalRegistry,
) -> list[PaperExtractionResult]:
    """Parse extract_entities tool_use blocks from one response into results."""
    results: list[PaperExtractionResult] = []
    for block in blocks:
        if block.type != "tool_use" or block.name != "extract_entities":
            continue

        inp = block.input
        pmid = str(inp.get("pmid", ""))
        if not pmid or pmid not in paper_by_pmid:
            _logger.warning(f"Extracted PMID {pmid!r} not in request — skipping")
            continue

        paper = paper_by_pmid[pmid]
        entities = _parse_entities(inp.get("entities", []), pmid, registry)
        relationships = _parse_relationships(inp.get("relationships", []), pmid, registry)

        results.append(
            PaperExtractionResult(pmid=pmid, entities=entities, relationships=relationships)
        )
        paper.entity_names = [e.canonical_id for e in entities]
        _logger.info(f"PMID {pmid}: {len(entities)} entities, {len(relationships)} relationships")

    return results


def _write_results(
    out_f,
    results: list[PaperExtractionResult],
    done_pmids: set[str],
    progress_path: Path,
    registry: CanonicalRegistry,
) -> None:
    """Append results to the output file and advance the resumability trackers."""
    if not results:
        return
    for result in results:
        out_f.write(json.dumps(result.to_dict()) + "\n")
        done_pmids.add(result.pmid)
    out_f.flush()
    _save_progress(progress_path, done_pmids)
    registry.save()


def _format_batch(batch: list[ALSPaper]) -> str:
    parts = [
        f"Extract entities from each of the following {len(batch)} ALS papers. "
        "Call extract_entities once per paper.\n"
    ]
    for paper in batch:
        text = paper.full_text if paper.full_text else paper.abstract
        # Cap at 2000 chars — 20-paper batches at ~500 tokens each stay well under 8192 output limit
        excerpt = text[:2000] if text else paper.abstract[:1000]
        parts.append(
            f"--- PMID:{paper.pmid} ---\n"
            f"Title: {paper.title}\n\n"
            f"{excerpt}\n"
        )
    return "\n".join(parts)


def _parse_entities(
    raw: list[dict],
    pmid: str,
    registry: CanonicalRegistry,
) -> list[ExtractedEntity]:
    entities = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "").strip()
        entity_type = item.get("type", "").strip()
        if not name or not entity_type:
            continue
        canonical_id = registry.resolve(name, entity_type)
        entities.append(
            ExtractedEntity(
                type=entity_type,
                name=name,
                canonical_id=canonical_id,
                confidence=float(item.get("confidence", 0.7)),
                mentions=int(item.get("mentions", 1)),
            )
        )
    return entities


def _parse_relationships(
    raw: list[dict],
    pmid: str,
    registry: CanonicalRegistry,
) -> list[EntityRelationship]:
    rels = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        source_name = item.get("source", "").strip()
        target_name = item.get("target", "").strip()
        rel_type = item.get("type", "").strip()
        if not source_name or not target_name or not rel_type:
            continue
        # We don't know entity types for source/target here — infer from name
        source_id = registry.resolve(source_name, _guess_type(source_name))
        target_id = registry.resolve(target_name, _guess_type(target_name))
        rels.append(
            EntityRelationship(
                source=source_id,
                target=target_id,
                relation_type=rel_type,
                evidence_pmids=[pmid],
                confidence=0.7,
                evidence_text=item.get("evidence_text", "")[:300],
            )
        )
    return rels


_guess_type = guess_entity_type


def _load_papers(path: Path) -> list[ALSPaper]:
    papers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(ALSPaper.from_dict(json.loads(line)))
    return papers


def _load_progress(path: Path) -> set[str]:
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def _save_progress(path: Path, done: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(done)))


def _load_batch_state(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_batch_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state))


def _clear_batch_state(path: Path) -> None:
    path.unlink(missing_ok=True)
