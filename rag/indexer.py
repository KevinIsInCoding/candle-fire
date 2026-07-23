"""ChromaDB collection builder with section-aware chunking."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from config import CHROMA_COLLECTION, CHROMA_DIR, ENTITIES_PATH, PAPERS_PATH
from logging_config import get_logger
from models import ALSPaper

_logger = get_logger("rag.indexer")

# Use MPS on Apple Silicon, CUDA on NVIDIA, otherwise CPU.
_DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# BioLORD-2023-C: anchored to UMLS/SNOMED CT/MeSH ontologies — natively understands
# biomedical synonyms (TARDBP = TDP-43, SOD1 = superoxide dismutase) and clinical phrasing.
_EMBED_FN = SentenceTransformerEmbeddingFunction(model_name="FremyCompany/BioLORD-2023-C", device=_DEVICE)


def _chunk_paper(paper: ALSPaper) -> list[dict]:
    """
    Split a paper into indexable chunks.
    - Full text available: one chunk per section (split on [Section Title] markers)
    - Abstract only: single chunk = title + mesh terms + abstract
    """
    base_meta = {
        "pmid": paper.pmid,
        "title": paper.title,
        "year": paper.year,
        "doi": paper.doi,
        "citation_count": paper.citation_count,
        # ChromaDB metadata must be scalar — serialize lists as comma-separated strings
        "entity_names": ",".join(paper.entity_names),
        "mesh_terms": ",".join(paper.mesh_terms[:10]),  # cap to avoid huge metadata
        "has_full_text": int(bool(paper.full_text)),  # bool not supported → int
    }

    if paper.full_text:
        sections: list[tuple[str, str]] = []
        current_title = "Abstract"
        current_lines: list[str] = [paper.abstract]

        for line in paper.full_text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]") and len(stripped) < 80:
                if current_lines:
                    body = "\n".join(current_lines).strip()
                    if body:
                        sections.append((current_title, body))
                current_title = stripped[1:-1]
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_title, body))

        # Prioritise high-value sections; cap at 6 total to keep index lean.
        # With 500 papers the cross-encoder only sees 20 candidates anyway —
        # 50+ chunks per paper adds noise without improving recall.
        _PRIORITY = {"abstract", "introduction", "results", "discussion", "conclusion", "methods"}
        priority = [s for s in sections if s[0].lower() in _PRIORITY]
        others = [s for s in sections if s[0].lower() not in _PRIORITY]
        selected = (priority + others)[:6]

        chunks = []
        for i, (section_title, section_text) in enumerate(selected):
            doc = f"{paper.title}\n[{section_title}]\n{section_text}"
            chunks.append({
                "id": f"{paper.pmid}_s{i}",
                "document": doc,
                "metadata": {**base_meta, "section": section_title, "chunk_index": i},
            })
        return chunks if chunks else [_abstract_chunk(paper, base_meta)]

    return [_abstract_chunk(paper, base_meta)]


def _abstract_chunk(paper: ALSPaper, base_meta: dict) -> dict:
    doc = f"{paper.title}\n{' '.join(paper.mesh_terms)}\n{paper.abstract}"
    return {
        "id": paper.pmid,
        "document": doc,
        "metadata": {**base_meta, "section": "abstract", "chunk_index": 0},
    }


def _load_entity_map(entities_path: Path = ENTITIES_PATH) -> dict[str, list[str]]:
    """Build pmid → [canonical_id, ...] map from entities.jsonl, if it exists."""
    if not entities_path.exists():
        return {}
    entity_map: dict[str, list[str]] = {}
    with open(entities_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pmid = str(r.get("pmid", ""))
            if pmid:
                entity_map[pmid] = [
                    e["canonical_id"] for e in r.get("entities", []) if e.get("canonical_id")
                ]
    return entity_map


def build_collection(
    papers_path: Path = PAPERS_PATH,
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
    reset: bool = False,
) -> chromadb.Collection:
    """Build ChromaDB collection from papers.jsonl. Idempotent — skips already-indexed chunks."""
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if reset:
        try:
            client.delete_collection(collection_name)
            _logger.info(f"Deleted collection: {collection_name}")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=_EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )

    entity_map = _load_entity_map()
    _logger.info(f"Entity map loaded: {len(entity_map)} PMIDs with extracted entities")

    papers: list[ALSPaper] = []
    with open(papers_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                paper = ALSPaper.from_dict(json.loads(line))
                paper.entity_names = entity_map.get(str(paper.pmid), paper.entity_names)
                papers.append(paper)

    _logger.info(f"Loaded {len(papers)} papers")

    all_chunks = []
    for paper in papers:
        all_chunks.extend(_chunk_paper(paper))

    # Skip already-indexed chunks (safe to re-run)
    existing_ids = set(collection.get(include=[])["ids"])
    new_chunks = [c for c in all_chunks if c["id"] not in existing_ids]

    if not new_chunks:
        _logger.info("All chunks already indexed")
        return collection

    _logger.info(f"Indexing {len(new_chunks)} new chunks from {len(papers)} papers")

    batch_size = 100
    batches = [new_chunks[i : i + batch_size] for i in range(0, len(new_chunks), batch_size)]

    with Progress(
        TextColumn("[cyan]Embedding chunks[/cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("", total=len(new_chunks))
        for batch in batches:
            collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["document"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )
            progress.advance(task, len(batch))

    _logger.info(f"Collection '{collection_name}': {collection.count()} total chunks")
    return collection


def load_collection(
    chroma_dir: Path = CHROMA_DIR,
    collection_name: str = CHROMA_COLLECTION,
) -> chromadb.Collection:
    """Load an existing collection at query time (fast, no re-embedding)."""
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_collection(name=collection_name, embedding_function=_EMBED_FN)
