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

FIND_TRIALS_BY_LOCATION_TOOL: anthropic.types.ToolParam = {
    "name": "find_trials_by_location",
    "description": (
        "Find ALS clinical trials by the facility/institution or the geography (state, city, "
        "or country) where they are conducted. Use this when the physician's question names a "
        "hospital or research site (e.g. 'trials at Mass General Hospital') or a place (e.g. "
        "'trials in NY'). Returns matching trials enriched with recruiting status, an evidence-"
        "strength tier, key supporting papers, and related trials for the same compound."
    ),
    "input_schema": _load("find_trials_by_location"),
}

EXTRACT_TRIAL_TARGETS_TOOL: anthropic.types.ToolParam = {
    "name": "extract_trial_targets",
    "description": (
        "Extract the primary biological target(s) of an ALS clinical trial — the gene, protein, "
        "compound, or mechanism being tested or modulated. Call once per trial."
    ),
    "input_schema": _load("extract_trial_targets"),
}

SUMMARIZE_TRIAL_MECHANISM_TOOL: anthropic.types.ToolParam = {
    "name": "summarize_trial_mechanism",
    "description": (
        "Produce a RAG-grounded mechanism summary for one ALS clinical trial — its compound, "
        "targeting mechanism, animal/preclinical results, and original indication if repurposed. "
        "Animal results and repurposed-from claims must cite a provided evidence PMID or be "
        "reported as 'unknown'. Call once per trial."
    ),
    "input_schema": _load("summarize_trial_mechanism"),
}

CLASSIFY_THERAPY_TOOL: anthropic.types.ToolParam = {
    "name": "classify_therapy",
    "description": (
        "Classify one experimental ALS therapy: its canonical name, modality, molecular target, "
        "mechanism of action, and best-fit mechanism class from the ALS taxonomy. "
        "Call once per therapy, echoing back the therapy_key verbatim."
    ),
    "input_schema": _load("classify_therapy"),
}

EXTRACTION_TOOLS: list[anthropic.types.ToolParam] = [EXTRACT_ENTITIES_TOOL]
TRIAL_EXTRACTION_TOOLS: list[anthropic.types.ToolParam] = [EXTRACT_TRIAL_TARGETS_TOOL]
TRIAL_SUMMARY_TOOLS: list[anthropic.types.ToolParam] = [SUMMARIZE_TRIAL_MECHANISM_TOOL]
RESEARCH_TOOLS: list[anthropic.types.ToolParam] = [SEARCH_LANDSCAPE_TOOL, FIND_TRIALS_BY_LOCATION_TOOL]
LANDSCAPE_TOOLS: list[anthropic.types.ToolParam] = [CLASSIFY_THERAPY_TOOL]
