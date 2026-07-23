"""Graph traversal: entity expansion, trial lookup, evidence retrieval."""
from __future__ import annotations

import networkx as nx

from config import KG_EXPANSION_HOPS, KG_MIN_EDGE_CONFIDENCE
from extraction.normalizer import normalize_entity, _GENE_ALIASES, _COMPOUND_ALIASES
from logging_config import get_logger

_logger = get_logger("graph.query")


def expand_query_entities(
    G: nx.DiGraph,
    entity_names: list[str],
    max_hops: int = KG_EXPANSION_HOPS,
) -> list[str]:
    """
    Map entity names to graph node IDs, then BFS-expand up to max_hops.
    Returns a deduplicated list of entity display_names for RAG search.
    Only traverses edges above KG_MIN_EDGE_CONFIDENCE.
    """
    if not G or not entity_names:
        return entity_names

    # Phase 1: map names to canonical node IDs
    seed_nodes: set[str] = set()
    for name in entity_names:
        matched = _find_node(G, name)
        if matched:
            seed_nodes.update(matched)
        else:
            _logger.debug(f"No graph node for query entity: {name!r}")

    if not seed_nodes:
        return entity_names  # fall through to lexical RAG search

    # Phase 2: BFS expansion
    frontier = set(seed_nodes)
    expanded = set(seed_nodes)

    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for node in frontier:
            for neighbor in list(G.successors(node)) + list(G.predecessors(node)):
                if neighbor in expanded:
                    continue
                # Only follow confident edges
                edge_data = G.get_edge_data(node, neighbor) or G.get_edge_data(neighbor, node) or {}
                if edge_data.get("confidence", 1.0) >= KG_MIN_EDGE_CONFIDENCE:
                    # Skip trial nodes — they inflate retrieval noise
                    if G.nodes[neighbor].get("type") != "ClinicalTrial":
                        next_frontier.add(neighbor)
        expanded.update(next_frontier)
        frontier = next_frontier

    # Phase 3: convert canonical IDs back to display names for RAG text queries
    display_names: list[str] = []
    seen: set[str] = set()
    for node_id in expanded:
        if node_id.startswith("trial:"):
            continue
        name = G.nodes[node_id].get("display_name", node_id.split(":", 1)[-1])
        if name not in seen:
            display_names.append(name)
            seen.add(name)

    _logger.info(
        f"KG expansion: {len(entity_names)} query entities → {len(display_names)} expanded",
        extra={"data": {"seeds": list(seed_nodes), "expanded_count": len(expanded)}},
    )
    return display_names


_STATUS_RANK = {"RECRUITING": 0, "ACTIVE_NOT_RECRUITING": 1, "NOT_YET_RECRUITING": 2, "COMPLETED": 3}


def find_trials_for_entities(
    G: nx.DiGraph,
    entity_names: list[str],
    max_trials: int = 10,
) -> list[dict]:
    """Return clinical trials linked to the given entity names.

    Collects all matches, scores by number of linked entities, sorts by
    status (RECRUITING first) then score, and returns the top max_trials.
    """
    if not G or not entity_names:
        return []

    target_nodes: set[str] = set()
    for name in entity_names:
        matched = _find_node(G, name)
        target_nodes.update(matched)

    # score[nct_id] = number of query entities this trial links to
    scores: dict[str, int] = {}
    meta: dict[str, dict] = {}

    for node_id in target_nodes:
        for pred in G.predecessors(node_id):
            if not pred.startswith("trial:"):
                continue
            nct_id = G.nodes[pred].get("nct_id", "")
            if not nct_id:
                continue
            scores[nct_id] = scores.get(nct_id, 0) + 1
            if nct_id not in meta:
                status = G.nodes[pred].get("status", "")
                meta[nct_id] = {
                    "nct_id": nct_id,
                    "title": G.nodes[pred].get("display_name", ""),
                    "phase": G.nodes[pred].get("phase", ""),
                    "status": status,
                    "url": G.nodes[pred].get("url", ""),
                    "_status_rank": _STATUS_RANK.get(status, 9),
                }

    ranked = sorted(
        meta.values(),
        key=lambda t: (t["_status_rank"], -scores[t["nct_id"]]),
    )
    for t in ranked:
        del t["_status_rank"]

    return ranked[:max_trials]


def get_entity_evidence(G: nx.DiGraph, canonical_id: str) -> dict:
    """Return node attributes + connected entity names for a canonical ID."""
    if not G.has_node(canonical_id):
        return {}
    attrs = dict(G.nodes[canonical_id])
    attrs["neighbors"] = [
        {
            "id": n,
            "display_name": G.nodes[n].get("display_name", n),
            "relation": G.get_edge_data(canonical_id, n, {}).get("relation_type", ""),
        }
        for n in G.successors(canonical_id)
        if G.nodes[n].get("type") != "ClinicalTrial"
    ]
    return attrs


def _find_node(G: nx.DiGraph, name: str) -> list[str]:
    """Map a raw entity name to zero or more graph node IDs."""
    hits: list[str] = []

    # 1. Try each entity type prefix
    for etype in ("Gene", "Protein", "Compound", "Mechanism", "Pathway", "Phenotype"):
        candidate = normalize_entity(name, etype)
        if G.has_node(candidate):
            hits.append(candidate)

    if hits:
        return hits

    # 2. Case-insensitive display_name match
    name_lower = name.lower()
    for node_id, data in G.nodes(data=True):
        display = data.get("display_name", "").lower()
        if display == name_lower or name_lower in display:
            hits.append(node_id)

    return hits
