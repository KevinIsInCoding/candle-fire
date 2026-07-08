"""Save and load the ALS knowledge graph (pickle for speed, JSON for inspection)."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import networkx as nx

from config import GRAPH_JSON_PATH, GRAPH_PICKLE_PATH
from logging_config import get_logger

_logger = get_logger("graph.serializer")


def save_graph(
    G: nx.DiGraph,
    pickle_path: Path = GRAPH_PICKLE_PATH,
    json_path: Path = GRAPH_JSON_PATH,
) -> None:
    for path in (pickle_path, json_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(pickle_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    data = {
        "nodes": [
            {"id": n, **{k: _json_safe(v) for k, v in G.nodes[n].items()}}
            for n in G.nodes
        ],
        "edges": [
            {"source": u, "target": v, **{k: _json_safe(dv) for k, dv in d.items()}}
            for u, v, d in G.edges(data=True)
        ],
        "stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        },
    }
    json_path.write_text(json.dumps(data, indent=2))
    _logger.info(f"Saved graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


def load_graph(pickle_path: Path = GRAPH_PICKLE_PATH) -> nx.DiGraph:
    if not pickle_path.exists():
        raise FileNotFoundError(
            f"Graph not found at {pickle_path}. "
            "Run: uv run python scripts/build_graph.py"
        )
    with open(pickle_path, "rb") as f:
        G = pickle.load(f)
    _logger.info(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def _json_safe(value: object) -> object:
    if isinstance(value, set):
        return list(value)
    if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
        return value
    return str(value)
