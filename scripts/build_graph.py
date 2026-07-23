#!/usr/bin/env python3
"""
Build the ALS knowledge graph from extracted entities + trials.
Run after scripts/extract_entities.py.

Usage:
    uv run python scripts/build_graph.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console

from config import ENTITIES_PATH, GRAPH_JSON_PATH, GRAPH_PICKLE_PATH, TRIALS_PATH
from graph.builder import build_graph
from graph.serializer import save_graph

console = Console()


def main() -> None:
    console.print("[cyan]Building ALS knowledge graph...[/cyan]")

    if not ENTITIES_PATH.exists():
        console.print(
            "[yellow]No entities.jsonl found — graph will be seeded only (no NER enrichment).[/yellow]"
        )
        console.print("[dim]Run first: uv run python scripts/extract_entities.py[/dim]\n")

    G = build_graph(entities_path=ENTITIES_PATH, trials_path=TRIALS_PATH)
    save_graph(G, pickle_path=GRAPH_PICKLE_PATH, json_path=GRAPH_JSON_PATH)

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Nodes: [bold]{G.number_of_nodes()}[/bold]")
    console.print(f"  Edges: [bold]{G.number_of_edges()}[/bold]")
    console.print(f"  Pickle: {GRAPH_PICKLE_PATH}")
    console.print(f"  JSON:   {GRAPH_JSON_PATH}")

    top = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
    if top:
        console.print("\n[dim]Most connected nodes:[/dim]")
        for node_id, deg in top:
            name = G.nodes[node_id].get("display_name", node_id)
            console.print(f"  [dim]{name} ({G.nodes[node_id].get('type', '?')}) — {deg} edges[/dim]")

    console.print("\nTest a query with KG expansion:")
    console.print("  [dim]uv run python main.py \"What is the role of C9orf72 in ALS?\"[/dim]")


if __name__ == "__main__":
    main()
