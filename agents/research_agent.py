"""Multi-step ALS research synthesis agent (streaming). Mirrors beacon/agents/research.py pattern."""
from __future__ import annotations

import json
from collections.abc import Generator

import anthropic
import chromadb
import networkx as nx
from sentence_transformers import CrossEncoder

from config import (
    CROSS_ENCODER_MODEL,
    CROSS_ENCODER_TOP_N,
    RETRIEVAL_ENTITY_N,
    RETRIEVAL_SEMANTIC_N,
    RRF_K,
    RRF_TOP_N,
    SYNTHESIS_MODEL,
)
from agents.grounding import enforce_grounding
from graph import query as kg_query
from llm import cached_system, cached_tools
from logging_config import get_logger
from prompts import SYNTHESIS_SYSTEM
from rag import retriever as rag_retriever
from tools import RESEARCH_TOOLS

_logger = get_logger("agents.research_agent")

# Loaded once at startup — ~80MB model, ~80ms/pair on CPU
_cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)


def _gate(answer: str, pmid_excerpts: dict[str, str]) -> str:
    """Apply the grounding gate to the final answer, logging any bullets removed."""
    result = enforce_grounding(answer, pmid_excerpts)
    if result.changed:
        _logger.warning(
            "grounding gate removed unsupported claims",
            extra={"data": {"removed_count": len(result.removed), "removed": result.removed}},
        )
    return result.text


def stream_research_agent(
    client: anthropic.Anthropic,
    query: str,
    collection: chromadb.Collection,
    trials: list[dict],
    graph: nx.DiGraph | None = None,
) -> Generator[tuple[str, str], None, None]:
    """
    Stream the ALS research synthesis agent.

    Yields:
        ("token", str)   — partial text chunk for streaming display
        ("status", str)  — status message during tool execution
        ("done", str)    — final complete response text
    """
    messages: list[anthropic.types.MessageParam] = [
        {"role": "user", "content": query}
    ]
    # Attach graph reference so _handle_search can use KG expansion
    _graph = graph

    # PMID -> retrieved excerpt, accumulated across every search the agent runs.
    # The grounding gate checks the model's inline (PMID: N) citations against these.
    pmid_excerpts: dict[str, str] = {}

    while True:
        stream_text = ""

        with client.messages.stream(
            model=SYNTHESIS_MODEL,
            max_tokens=4096,
            system=cached_system(SYNTHESIS_SYSTEM),
            tools=cached_tools(RESEARCH_TOOLS),
            messages=messages,
        ) as stream:
            # Accumulate tool-use input JSON alongside streaming text
            tool_calls: list[dict] = []
            current_tool: dict | None = None
            current_input_json = ""

            for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "tool_use":
                        current_tool = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                        }
                        current_input_json = ""
                        yield ("status", "Searching knowledge base and re-ranking results for precision...")

                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        chunk = event.delta.text
                        stream_text += chunk
                        yield ("token", chunk)
                    elif event.delta.type == "input_json_delta" and current_tool:
                        current_input_json += event.delta.partial_json

                elif event.type == "content_block_stop":
                    if current_tool is not None:
                        try:
                            current_tool["input"] = json.loads(current_input_json)
                        except json.JSONDecodeError:
                            current_tool["input"] = {}
                        tool_calls.append(current_tool)
                        current_tool = None
                        current_input_json = ""

            final_msg = stream.get_final_message()

        messages.append({"role": "assistant", "content": final_msg.content})

        if final_msg.stop_reason == "end_turn":
            yield ("done", _gate(stream_text, pmid_excerpts))
            return

        if final_msg.stop_reason == "tool_use" and tool_calls:
            tool_results: list[anthropic.types.ToolResultBlockParam] = []

            for tool_call in tool_calls:
                if tool_call["name"] == "search_research_landscape":
                    result = _handle_search(tool_call["input"], collection, trials, _graph)
                    is_error = False
                    for paper in result.get("papers", []):
                        pmid = str(paper.get("pmid", ""))
                        if pmid and pmid not in pmid_excerpts:
                            pmid_excerpts[pmid] = paper.get("excerpt", "")
                else:
                    result = {"error": f"Unknown tool: {tool_call['name']}"}
                    is_error = True

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": json.dumps(result),
                    "is_error": is_error,
                })

            messages.append({"role": "user", "content": tool_results})
        else:
            yield ("done", _gate(stream_text, pmid_excerpts))
            return


def _handle_search(
    tool_input: dict,
    collection: chromadb.Collection,
    trials: list[dict],
    graph: nx.DiGraph | None = None,
) -> dict:
    """Execute KG expansion → RAG search → trial lookup and return structured context."""
    query_text = tool_input.get("query_text", "")
    query_entities = tool_input.get("query_entities", [])

    # Step 1: KG expansion — surface related entities Claude didn't name explicitly
    # e.g. "tofersen" → expands to ["SOD1", "antisense oligonucleotide", "RNA splicing"]
    if graph and query_entities:
        expanded_entities = kg_query.expand_query_entities(graph, query_entities)
    else:
        expanded_entities = query_entities

    # Step 2: Semantic search → up to 30 papers (pure similarity, no citation weight yet)
    semantic_results = rag_retriever.search(collection, query_text, n_results=RETRIEVAL_SEMANTIC_N)

    # Step 3: Entity-targeted search → up to 30 papers (one query per expanded entity)
    entity_results = rag_retriever.search_by_entities(
        collection, expanded_entities, n_results=RETRIEVAL_ENTITY_N
    )

    # Step 4: RRF merge → top 20 papers
    merged = rag_retriever.rrf_merge(
        [semantic_results, entity_results], k=RRF_K, top_n=RRF_TOP_N
    )

    # Step 5: Cross-encoder rerank → top 15 papers
    reranked = rag_retriever.cross_encoder_rerank(
        _cross_encoder, query_text, merged, top_n=CROSS_ENCODER_TOP_N
    )

    # Step 6: Citation boost — final score = ce_score × log(citation_count + 2)
    top_papers = rag_retriever.apply_citation_boost(reranked)

    _logger.info(
        "KG+RAG+CE search",
        extra={"data": {
            "query_entities": query_entities,
            "expanded_entities": len(expanded_entities),
            "semantic_hits": len(semantic_results),
            "entity_hits": len(entity_results),
            "rrf_merged": len(merged),
            "after_cross_encoder": len(top_papers),
            "kg_active": graph is not None,
        }},
    )

    # Step 7: Trial matching — prefer KG-linked trials, fall back to text match
    related_trials: list[dict] = []
    if graph and query_entities:
        related_trials = kg_query.find_trials_for_entities(graph, expanded_entities, max_trials=10)

    if not related_trials:
        # Exact NCT ID match first — handles "NCT06351592" style queries
        nct_ids_in_query = {
            w.upper() for w in query_text.split() if w.upper().startswith("NCT")
        }
        trial_by_nct = {t.get("nct_id", "").upper(): t for t in trials}
        for nct_id in nct_ids_in_query:
            if nct_id in trial_by_nct:
                t = trial_by_nct[nct_id]
                related_trials.append({
                    "nct_id": t.get("nct_id", ""),
                    "title": t.get("title", ""),
                    "phase": t.get("phase", ""),
                    "status": t.get("status", ""),
                    "url": t.get("url", ""),
                })

    if not related_trials and query_entities:
        entities_lower = [e.lower() for e in expanded_entities]
        for trial in trials:
            iv_names = " ".join(iv.get("name", "") for iv in trial.get("interventions", []))
            trial_text = f"{trial.get('title', '')} {iv_names}".lower()
            if any(e in trial_text for e in entities_lower):
                related_trials.append({
                    "nct_id": trial.get("nct_id", ""),
                    "title": trial.get("title", ""),
                    "phase": trial.get("phase", ""),
                    "status": trial.get("status", ""),
                    "url": trial.get("url", ""),
                })
            if len(related_trials) >= 5:
                break

    return {
        "papers": [
            {
                "pmid": r["pmid"],
                "title": r["title"],
                "year": r["year"],
                "doi": r["doi"],
                "citation_count": r["citation_count"],
                "section": r["section"],
                "excerpt": r["document"][:600],
                "score": round(r["score"], 3),
            }
            for r in top_papers
        ],
        "query_entities": query_entities,
        "expanded_entities": expanded_entities,
        "trials": related_trials,
        "evidence_count": len(top_papers),
        "kg_expansion_active": graph is not None,
    }
