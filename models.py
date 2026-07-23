from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ALSPaper:
    pmid: str
    title: str
    abstract: str
    authors: list[str]
    year: int
    doi: str
    mesh_terms: list[str]
    # Populated after PMC fetch (None if not Open Access)
    full_text: Optional[str] = None
    # Populated after Semantic Scholar enrichment
    citation_count: int = 0
    # Populated after entity extraction
    entity_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "year": self.year,
            "doi": self.doi,
            "mesh_terms": self.mesh_terms,
            "full_text": self.full_text,
            "citation_count": self.citation_count,
            "entity_names": self.entity_names,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ALSPaper:
        return cls(
            pmid=d["pmid"],
            title=d["title"],
            abstract=d["abstract"],
            authors=d.get("authors", []),
            year=d.get("year", 0),
            doi=d.get("doi", ""),
            mesh_terms=d.get("mesh_terms", []),
            full_text=d.get("full_text"),
            citation_count=d.get("citation_count", 0),
            entity_names=d.get("entity_names", []),
        )


@dataclass
class ExtractedEntity:
    type: str  # Gene | Protein | Compound | Pathway | Phenotype | Mechanism
    name: str  # raw name from text
    canonical_id: str  # normalized canonical identifier
    confidence: float  # 0.0–1.0
    mentions: int  # occurrence count in paper


@dataclass
class EntityRelationship:
    source: str  # canonical_id of source entity
    target: str  # canonical_id of target entity
    relation_type: str  # BINDS | INHIBITS | ASSOCIATED_WITH | TESTED_IN | EXPRESSED_IN | CO_OCCURS
    evidence_pmids: list[str]
    confidence: float  # average confidence across supporting papers
    evidence_text: str = ""  # representative excerpt from the paper


@dataclass
class PaperExtractionResult:
    pmid: str
    entities: list[ExtractedEntity]
    relationships: list[EntityRelationship]

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid,
            "entities": [
                {
                    "type": e.type,
                    "name": e.name,
                    "canonical_id": e.canonical_id,
                    "confidence": e.confidence,
                    "mentions": e.mentions,
                }
                for e in self.entities
            ],
            "relationships": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation_type": r.relation_type,
                    "evidence_pmids": r.evidence_pmids,
                    "confidence": r.confidence,
                    "evidence_text": r.evidence_text,
                }
                for r in self.relationships
            ],
        }


@dataclass
class TrialSummary:
    nct_id: str
    title: str
    phase: str
    status: str
    interventions: list[str]
    target_entities: list[str]  # canonical_ids of targeted genes/proteins/compounds
    sponsor: str = ""
    start_date: str = ""
    url: str = ""


@dataclass
class ResearchLandscape:
    query: str
    mechanisms: list[str]  # 2–3 key mechanism bullet points
    entities: list[dict]  # {name, type, description, paper_count}
    papers: list[dict]  # top 5: {pmid, title, year, doi, citation_count}
    trials: list[TrialSummary]
    evidence_count: int  # total papers retrieved
    generated_at: str  # ISO timestamp
