from __future__ import annotations

import json
from pathlib import Path

import anthropic

_DATA = Path(__file__).parent / "data" / "tools"


def _load(name: str) -> dict:
    return json.loads((_DATA / f"{name}.json").read_text())


EXTRACT_ENTITIES_TOOL: anthropic.types.ToolParam = {
    "name": "extract_entities",
    "description": (
        "Extract biomedical entities and relationships from an ALS paper abstract. "
        "For each entity, identify its type (Gene, Protein, Compound, Pathway, Phenotype, or Mechanism), "
        "the exact name as it appears in the text, and the confidence of the identification. "
        "For relationships, identify the source entity, target entity, and the type of relationship."
    ),
    "input_schema": _load("extract_entities"),
}

SEARCH_LANDSCAPE_TOOL: anthropic.types.ToolParam = {
    "name": "search_research_landscape",
    "description": (
        "Search the ALS research knowledge base by combining knowledge graph traversal and "
        "vector similarity search. Provide the entities you identified in the physician's query "
        "and the original query text. Returns ranked papers, related biological entities, "
        "and linked clinical trials."
    ),
    "input_schema": _load("search_landscape"),
}

EXTRACT_TRIAL_TARGETS_TOOL: anthropic.types.ToolParam = {
    "name": "extract_trial_targets",
    "description": (
        "Extract the primary biological target(s) of an ALS clinical trial — the gene, protein, "
        "compound, or mechanism being tested or modulated. Call once per trial."
    ),
    "input_schema": _load("extract_trial_targets"),
}

EXTRACTION_TOOLS: list[anthropic.types.ToolParam] = [EXTRACT_ENTITIES_TOOL]
TRIAL_EXTRACTION_TOOLS: list[anthropic.types.ToolParam] = [EXTRACT_TRIAL_TARGETS_TOOL]
RESEARCH_TOOLS: list[anthropic.types.ToolParam] = [SEARCH_LANDSCAPE_TOOL]
