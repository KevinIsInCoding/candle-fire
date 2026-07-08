"""Multi-step ALS research synthesis agent (streaming). Mirrors beacon/agents/research.py pattern."""
from __future__ import annotations

import json
from collections.abc import Generator

import anthropic
import chromadb

from config import SYNTHESIS_MODEL
from llm import cached_system, cached_tools
from logging_config import get_logger
from prompts import SYNTHESIS_SYSTEM
from rag import retriever as rag_retriever
from tools import RESEARCH_TOOLS

_logger = get_logger("agents.research_agent")


def stream_research_agent(
    client: anthropic.Anthropic,
    query: str,
    collection: chromadb.Collection,
    trials: list[dict],
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
                        yield ("status", "Searching ALS research knowledge base...")

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
            yield ("done", stream_text)
            return

        if final_msg.stop_reason == "tool_use" and tool_calls:
            tool_results: list[anthropic.types.ToolResultBlockParam] = []

            for tool_call in tool_calls:
                if tool_call["name"] == "search_research_landscape":
                    result = _handle_search(tool_call["input"], collection, trials)
                    is_error = False
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
            yield ("done", stream_text)
            return


def _handle_search(
    tool_input: dict,
    collection: chromadb.Collection,
    trials: list[dict],
) -> dict:
    """Execute RAG search + trial lookup and return structured context for Claude."""
    query_text = tool_input.get("query_text", "")
    query_entities = tool_input.get("query_entities", [])

    # Semantic search
    semantic_results = rag_retriever.search(collection, query_text, n_results=10)

    # Entity-targeted search (finds papers even if query terms don't match directly)
    entity_results = rag_retriever.search_by_entities(collection, query_entities, n_results=15)

    # Merge by PMID — keep best score per paper
    seen: dict[str, dict] = {}
    for r in semantic_results + entity_results:
        pmid = r["pmid"]
        if pmid not in seen or r["score"] > seen[pmid]["score"]:
            seen[pmid] = r

    top_papers = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:15]

    _logger.info(
        "RAG search",
        extra={"data": {
            "query_entities": query_entities,
            "semantic_hits": len(semantic_results),
            "entity_hits": len(entity_results),
            "merged": len(top_papers),
        }},
    )

    # Match trials by entity name in title or intervention text
    related_trials = []
    if query_entities:
        entities_lower = [e.lower() for e in query_entities]
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
        "trials": related_trials,
        "evidence_count": len(top_papers),
    }
