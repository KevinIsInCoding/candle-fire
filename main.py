#!/usr/bin/env python3
"""CLI interface for candle-fire ALS research synthesis."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import anthropic
from rich.console import Console

from agents.research_agent import stream_research_agent
from config import CHROMA_COLLECTION, CHROMA_DIR, TRIALS_PATH

console = Console()

_EXAMPLES = [
    "What's the evidence for tofersen targeting SOD1 in ALS?",
    "What mechanisms link TDP-43 to ALS pathology?",
    "What compounds target glutamate excitotoxicity in ALS?",
    "What is the role of C9orf72 repeat expansion in ALS?",
]


def _load_trials() -> list[dict]:
    if not TRIALS_PATH.exists():
        return []
    with open(TRIALS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    from rag.indexer import load_collection

    query = " ".join(sys.argv[1:]).strip()

    # Load resources
    console.print("[cyan]Loading knowledge base...[/cyan]", end="\r")
    try:
        collection = load_collection(CHROMA_DIR, CHROMA_COLLECTION)
    except Exception:
        console.print("[red]ChromaDB collection not found.[/red]")
        console.print("[yellow]Run: uv run python scripts/build_index.py[/yellow]")
        sys.exit(1)

    trials = _load_trials()
    client = anthropic.Anthropic()

    n_chunks = collection.count()
    console.print(
        f"[green]Ready.[/green] {n_chunks} chunks indexed · {len(trials)} trials loaded.{' ' * 20}"
    )

    if not query:
        console.print("\n[dim]Example questions:[/dim]")
        for ex in _EXAMPLES:
            console.print(f"[dim]  • {ex}[/dim]")
        console.print()
        try:
            query = console.input("[bold]Ask about ALS research:[/bold] ").strip()
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

    if not query:
        sys.exit(0)

    console.print()

    for event_type, content in stream_research_agent(client, query, collection, trials):
        if event_type == "status":
            console.print(f"[dim italic]{content}[/dim italic]")
        elif event_type == "token":
            console.print(content, end="", highlight=False)
        elif event_type == "done":
            console.print()


if __name__ == "__main__":
    main()
