"""Shared loading for the data suite.

Loads the gold questions plus the heavy runtime assets (graph pickle + ChromaDB
collection) exactly once, so the three data checks don't each pay the embedding
model + index open cost.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property

from config import CHROMA_COLLECTION, CHROMA_DIR, GRAPH_PICKLE_PATH
from evals import thresholds as T


def load_questions() -> list[dict]:
    return [json.loads(l) for l in T.QUESTIONS_GOLD.read_text().splitlines() if l.strip()]


def data_assets_present() -> bool:
    """True when the runtime data needed by the data suite exists locally."""
    return (CHROMA_DIR / "chroma.sqlite3").exists() and GRAPH_PICKLE_PATH.exists()


@dataclass
class DataContext:
    questions: list[dict]

    @cached_property
    def graph(self):
        from graph.serializer import load_graph
        return load_graph()

    @cached_property
    def collection(self):
        from rag.indexer import load_collection
        return load_collection(CHROMA_DIR, CHROMA_COLLECTION)
