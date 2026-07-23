"""
NetworkX DiGraph construction from extracted entities + trials.
Upserts nodes and edges (merges PMIDs, recomputes confidence average).
Pre-populates with ALS seed entities from config.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from config import (
    DERIVED_SEEDS_PATH,
    ENTITIES_PATH,
    KG_MIN_EDGE_CONFIDENCE,
    MANUAL_SEEDS_PATH,
    TRIALS_PATH,
)
from extraction.normalizer import normalize_entity
from logging_config import get_logger
from models import PaperExtractionResult

_logger = get_logger("graph.builder")


def build_graph(
    entities_path: Path = ENTITIES_PATH,
    trials_path: Path = TRIALS_PATH,
) -> nx.DiGraph:
    """Build the ALS knowledge graph. Returns a populated DiGraph."""
    G: nx.DiGraph = nx.DiGraph()

    _add_seed_entities(G)
    _logger.info(f"Seeded graph: {G.number_of_nodes()} seed nodes")

    if entities_path.exists():
        n_papers = _add_extracted_entities(G, entities_path)
        _logger.info(
            f"After extraction: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges from {n_papers} papers"
        )
    else:
        _logger.warning(f"Entities file not found: {entities_path} — skipping NER enrichment")

    if trials_path.exists():
        n_trials = _add_trials(G, trials_path)
        _logger.info(f"Added {n_trials} trial nodes")
    else:
        _logger.warning(f"Trials file not found: {trials_path}")

    return G


def _add_seed_entities(G: nx.DiGraph) -> None:
    type_map = {
        "genes": "Gene",
        "proteins": "Protein",
        "compounds": "Compound",
        "mechanisms": "Mechanism",
        "phenotypes": "Phenotype",
    }

    manual = json.loads(MANUAL_SEEDS_PATH.read_text())

    derived: dict[str, list[str]] = {}
    if DERIVED_SEEDS_PATH.exists():
        derived = json.loads(DERIVED_SEEDS_PATH.read_text())

    for category, entity_type in type_map.items():
        seen: set[str] = set()
        for name in manual.get(category, []):
            canonical_id = normalize_entity(name, entity_type)
            seen.add(canonical_id)
            _upsert_node(G, canonical_id, {
                "type": entity_type,
                "display_name": name,
                "paper_count": 0,
                "evidence_pmids": [],
                "is_seed": True,
            })
        for name in derived.get(category, []):
            canonical_id = normalize_entity(name, entity_type)
            if canonical_id in seen:
                continue
            seen.add(canonical_id)
            _upsert_node(G, canonical_id, {
                "type": entity_type,
                "display_name": name,
                "paper_count": 0,
                "evidence_pmids": [],
                "is_seed": False,
                "is_derived_seed": True,
            })


def _add_extracted_entities(G: nx.DiGraph, entities_path: Path) -> int:
    n_papers = 0
    with open(entities_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            pmid = raw["pmid"]
            n_papers += 1

            # Add / update entity nodes
            for ent in raw.get("entities", []):
                canonical_id = ent.get("canonical_id", "")
                if not canonical_id:
                    continue
                if G.has_node(canonical_id):
                    G.nodes[canonical_id]["paper_count"] += 1
                    G.nodes[canonical_id]["evidence_pmids"].append(pmid)
                    # Update confidence as running average
                    cur = G.nodes[canonical_id].get("confidence", 0.7)
                    G.nodes[canonical_id]["confidence"] = (cur + ent.get("confidence", 0.7)) / 2
                else:
                    _upsert_node(G, canonical_id, {
                        "type": ent.get("type", "Unknown"),
                        "display_name": ent.get("name", canonical_id),
                        "paper_count": 1,
                        "evidence_pmids": [pmid],
                        "confidence": ent.get("confidence", 0.7),
                        "is_seed": False,
                    })

            # Add / update relationship edges
            for rel in raw.get("relationships", []):
                source = rel.get("source", "")
                target = rel.get("target", "")
                rel_type = rel.get("relation_type", "")
                if not source or not target or not rel_type:
                    continue

                # Ensure both endpoints exist as nodes
                for node_id in (source, target):
                    if not G.has_node(node_id):
                        _upsert_node(G, node_id, {
                            "type": "Unknown",
                            "display_name": node_id.split(":", 1)[-1],
                            "paper_count": 0,
                            "evidence_pmids": [],
                            "is_seed": False,
                        })

                conf = rel.get("confidence", 0.7)
                if G.has_edge(source, target):
                    edge = G[source][target]
                    if rel_type not in edge.get("relation_types", []):
                        edge.setdefault("relation_types", [edge.get("relation_type", rel_type)])
                        edge["relation_types"].append(rel_type)
                    if pmid not in edge["evidence_pmids"]:
                        edge["evidence_pmids"].append(pmid)
                    # Running average confidence
                    edge["confidence"] = (edge["confidence"] + conf) / 2
                else:
                    G.add_edge(source, target, **{
                        "relation_type": rel_type,
                        "relation_types": [rel_type],
                        "evidence_pmids": [pmid],
                        "confidence": conf,
                        "evidence_text": rel.get("evidence_text", ""),
                    })

    return n_papers


def _add_trials(G: nx.DiGraph, trials_path: Path) -> int:
    n_trials = 0
    with open(trials_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trial = json.loads(line)
            nct_id = trial.get("nct_id", "")
            if not nct_id:
                continue

            node_id = f"trial:{nct_id}"
            _upsert_node(G, node_id, {
                "type": "ClinicalTrial",
                "display_name": trial.get("title", nct_id)[:120],
                "nct_id": nct_id,
                "phase": trial.get("phase", ""),
                "status": trial.get("status", ""),
                "url": trial.get("url", f"https://clinicaltrials.gov/study/{nct_id}"),
                "paper_count": 0,
                "evidence_pmids": [],
            })

            # Link trial to known target entities
            for target_name in trial.get("target_entities", []):
                # Try gene, compound, protein
                matched = False
                for etype in ("Gene", "Compound", "Protein", "Mechanism"):
                    candidate_id = normalize_entity(target_name, etype)
                    if G.has_node(candidate_id):
                        if not G.has_edge(node_id, candidate_id):
                            G.add_edge(node_id, candidate_id, **{
                                "relation_type": "TESTED_IN",
                                "relation_types": ["TESTED_IN"],
                                "evidence_pmids": [],
                                "confidence": 1.0,
                                "evidence_text": "",
                            })
                        matched = True
                        break
                if not matched:
                    # Novel entity from trial with no paper evidence — canonicalize as
                    # compound (trial targets are drugs) but type stays Unknown since
                    # we can't confirm mechanism/class without literature support.
                    fallback_id = normalize_entity(target_name, "Compound")
                    _upsert_node(G, fallback_id, {
                        "type": "Unknown",
                        "display_name": target_name,
                        "paper_count": 0,
                        "evidence_pmids": [],
                        "confidence": 0.0,
                        "is_seed": False,
                        "is_trial_derived": True,
                    })
                    G.add_edge(node_id, fallback_id, **{
                        "relation_type": "TESTED_IN",
                        "relation_types": ["TESTED_IN"],
                        "evidence_pmids": [],
                        "confidence": 1.0,
                        "evidence_text": "",
                    })
                    _logger.debug(f"Trial {nct_id}: created trial-derived node {fallback_id!r} (type=Unknown)")

            n_trials += 1
    return n_trials


def _upsert_node(G: nx.DiGraph, node_id: str, attrs: dict) -> None:
    if G.has_node(node_id):
        G.nodes[node_id].update(attrs)
    else:
        G.add_node(node_id, **attrs)
