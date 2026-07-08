"""
Claude Sonnet entity extractor.
Batches 10 papers per API call; resumable via .progress.json.
Uses full_text when available, otherwise abstract.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import anthropic
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from config import (
    ENTITIES_PATH,
    EXTRACTION_BATCH_SIZE,
    EXTRACTION_MODEL,
    EXTRACTION_PROGRESS_PATH,
    PAPERS_PATH,
)
from extraction.normalizer import CanonicalRegistry, normalize_entity
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
    """Extract entities from all papers. Skips already-processed PMIDs."""
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

    results: list[PaperExtractionResult] = []

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

        for i in range(0, len(pending), EXTRACTION_BATCH_SIZE):
            batch = pending[i : i + EXTRACTION_BATCH_SIZE]
            batch_results = _extract_batch(client, batch, registry)

            for result in batch_results:
                out_f.write(json.dumps(result.to_dict()) + "\n")
                done_pmids.add(result.pmid)
                results.append(result)

            _save_progress(progress_path, done_pmids)
            registry.save()
            progress.advance(task, len(batch))

            # Respect rate limits between batches
            if i + EXTRACTION_BATCH_SIZE < len(pending):
                time.sleep(1.0)

    return results


def _extract_batch(
    client: anthropic.Anthropic,
    batch: list[ALSPaper],
    registry: CanonicalRegistry,
) -> list[PaperExtractionResult]:
    """Send a batch of papers to Claude and collect one extract_entities call per paper."""
    user_content = _format_batch(batch)

    try:
        response = client.messages.create(
            model=EXTRACTION_MODEL,
            max_tokens=4096,
            system=_EXTRACTION_SYSTEM,
            tools=EXTRACTION_TOOLS,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_content}],
        )
    except anthropic.RateLimitError:
        _logger.warning("Rate limited — sleeping 30s")
        time.sleep(30)
        response = client.messages.create(
            model=EXTRACTION_MODEL,
            max_tokens=4096,
            system=_EXTRACTION_SYSTEM,
            tools=EXTRACTION_TOOLS,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": user_content}],
        )

    # Build a PMID→paper lookup so we can match extracted results back
    paper_by_pmid = {p.pmid: p for p in batch}

    results: list[PaperExtractionResult] = []
    for block in response.content:
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

    # For any paper with no Claude response, add an empty result so it's not re-processed
    found_pmids = {r.pmid for r in results}
    for p in batch:
        if p.pmid not in found_pmids:
            _logger.warning(f"No extraction result for PMID {p.pmid} — recording empty")
            results.append(PaperExtractionResult(pmid=p.pmid, entities=[], relationships=[]))

    return results


def _format_batch(batch: list[ALSPaper]) -> str:
    parts = [
        f"Extract entities from each of the following {len(batch)} ALS papers. "
        "Call extract_entities once per paper.\n"
    ]
    for paper in batch:
        text = paper.full_text if paper.full_text else paper.abstract
        # Cap at 3000 chars to stay within token budget for a 10-paper batch
        excerpt = text[:3000] if text else paper.abstract[:1000]
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


def _guess_type(name: str) -> str:
    """Best-effort entity type guess from name for relationship source/target."""
    from extraction.normalizer import _GENE_ALIASES, _COMPOUND_ALIASES
    if name.strip().upper() in _GENE_ALIASES or name.strip() in _GENE_ALIASES:
        return "Gene"
    if name.strip() in _COMPOUND_ALIASES:
        return "Compound"
    return "Protein"


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
