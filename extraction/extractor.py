"""
Claude Haiku entity extractor.
Batches 20 papers per API call; resumable via .progress.json.
Uses full_text when available, otherwise abstract.
Prompt caching on system + tools reduces per-call cost ~40%.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from config import (
    ENTITIES_PATH,
    EXTRACTION_BATCH_SIZE,
    EXTRACTION_MODEL,
    EXTRACTION_PROGRESS_PATH,
    EXTRACTION_WORKERS,
    PAPERS_PATH,
)
from extraction.normalizer import CanonicalRegistry, guess_entity_type, normalize_entity
from logging_config import get_logger
from models import ALSPaper, ExtractedEntity, EntityRelationship, PaperExtractionResult
from tools import EXTRACTION_TOOLS

_logger = get_logger("extraction.extractor")

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
    client: anthropic.Anthropic | None = None,
) -> list[PaperExtractionResult]:
    """Extract entities from all papers. Skips already-processed PMIDs.

    Runs EXTRACTION_WORKERS batches in parallel. A lock serializes file writes
    and progress saves so threads don't corrupt each other.
    """
    if client is None:
        client = anthropic.Anthropic()

    papers = _load_papers(papers_path)
    done_pmids = _load_progress(progress_path)

    pending = [p for p in papers if p.pmid not in done_pmids]
    _logger.info(f"{len(papers)} papers total; {len(done_pmids)} already processed; {len(pending)} pending")

    if not pending:
        return []

    registry = CanonicalRegistry()
    entities_path.parent.mkdir(parents=True, exist_ok=True)

    batches = [pending[i : i + EXTRACTION_BATCH_SIZE] for i in range(0, len(pending), EXTRACTION_BATCH_SIZE)]
    results: list[PaperExtractionResult] = []
    write_lock = threading.Lock()

    with (
        open(entities_path, "a", encoding="utf-8") as out_f,
        Progress(
            TextColumn("[cyan]{task.description}[/cyan]"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress,
    ):
        task = progress.add_task("Extracting entities", total=len(pending))

        def _process_batch(batch: list[ALSPaper]) -> list[PaperExtractionResult]:
            return _extract_batch(client, batch, registry)

        with ThreadPoolExecutor(max_workers=EXTRACTION_WORKERS) as pool:
            futures = {pool.submit(_process_batch, b): b for b in batches}
            for future in as_completed(futures):
                batch_results = future.result()
                with write_lock:
                    for result in batch_results:
                        out_f.write(json.dumps(result.to_dict()) + "\n")
                        done_pmids.add(result.pmid)
                        results.append(result)
                    out_f.flush()
                    _save_progress(progress_path, done_pmids)
                    registry.save()
                    progress.advance(task, len(futures[future]))

    return results


def _extract_batch(
    client: anthropic.Anthropic,
    batch: list[ALSPaper],
    registry: CanonicalRegistry,
) -> list[PaperExtractionResult]:
    """Send a batch of papers to Claude and collect one extract_entities call per paper."""
    paper_by_pmid = {p.pmid: p for p in batch}
    content_blocks = _call_claude(client, batch)

    results: list[PaperExtractionResult] = []
    for block in content_blocks:
        if block.type != "tool_use" or block.name != "extract_entities":
            continue

        inp = block.input
        pmid = str(inp.get("pmid", ""))
        if not pmid or pmid not in paper_by_pmid:
            _logger.warning(f"Extracted PMID {pmid!r} not in batch — skipping")
            continue

        paper = paper_by_pmid[pmid]
        entities = _parse_entities(inp.get("entities", []), pmid, registry)
        relationships = _parse_relationships(inp.get("relationships", []), pmid, registry)

        result = PaperExtractionResult(
            pmid=pmid,
            entities=entities,
            relationships=relationships,
        )
        results.append(result)
        _logger.info(f"PMID {pmid}: {len(entities)} entities, {len(relationships)} relationships")

        # Mark paper entity_names (used downstream by RAG indexer on re-index)
        paper.entity_names = [e.canonical_id for e in entities]

    # Retry any papers Claude missed — send them individually
    found_pmids = {r.pmid for r in results}
    missed = [p for p in batch if p.pmid not in found_pmids]
    if missed:
        _logger.info(f"Retrying {len(missed)} missed papers individually")
        for paper in missed:
            retry_results = _call_claude(client, [paper])
            for block in retry_results:
                if block.type != "tool_use" or block.name != "extract_entities":
                    continue
                inp = block.input
                pmid = str(inp.get("pmid", ""))
                if not pmid or pmid not in paper_by_pmid:
                    continue
                entities = _parse_entities(inp.get("entities", []), pmid, registry)
                relationships = _parse_relationships(inp.get("relationships", []), pmid, registry)
                results.append(PaperExtractionResult(pmid=pmid, entities=entities, relationships=relationships))
                paper_by_pmid[pmid].entity_names = [e.canonical_id for e in entities]
                found_pmids.add(pmid)
                _logger.info(f"Retry succeeded for PMID {pmid}")
            time.sleep(0.5)

    # Any still-missing after retry → record empty so they're not re-attempted
    for p in batch:
        if p.pmid not in found_pmids:
            _logger.warning(f"No extraction result for PMID {p.pmid} after retry — recording empty")
            results.append(PaperExtractionResult(pmid=p.pmid, entities=[], relationships=[]))

    return results


def _call_claude(client: anthropic.Anthropic, batch: list[ALSPaper]) -> list:
    """Raw Claude call — returns response.content blocks.

    Prompt caching: system and tools are static across all calls; adding
    cache_control to the last tool + system block caches the entire prefix
    (tools render before system in the API token order). Cache reads cost
    ~10% of normal input price, halving the effective per-call overhead.
    """
    # Cache the static system+tools prefix across batch calls
    cached_system = [{"type": "text", "text": _EXTRACTION_SYSTEM, "cache_control": {"type": "ephemeral"}}]
    cached_tools = list(EXTRACTION_TOOLS)
    if cached_tools:
        last = dict(cached_tools[-1])
        last["cache_control"] = {"type": "ephemeral"}
        cached_tools[-1] = last

    def _request() -> list:
        response = client.messages.create(
            model=EXTRACTION_MODEL,
            max_tokens=8192,
            system=cached_system,
            tools=cached_tools,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": _format_batch(batch)}],
        )
        return response.content

    try:
        return _request()
    except anthropic.RateLimitError:
        _logger.warning("Rate limited — sleeping 30s")
        time.sleep(30)
        return _request()


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
