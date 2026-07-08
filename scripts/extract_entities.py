#!/usr/bin/env python3
"""
Extract biomedical entities from ALS papers using Claude Sonnet.
Run after scripts/ingest_papers.py.

Usage:
    uv run python scripts/extract_entities.py
    uv run python scripts/extract_entities.py --max 50     # test with first 50 papers
    uv run python scripts/extract_entities.py --reset      # clear progress and re-extract
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import argparse
import json

import anthropic
from rich.console import Console

from config import (
    CANONICAL_IDS_PATH,
    ENTITIES_PATH,
    EXTRACTION_PROGRESS_PATH,
    PAPERS_PATH,
)
from extraction.extractor import extract_all

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract entities from ALS papers")
    parser.add_argument("--max", type=int, default=None, help="Only process first N papers")
    parser.add_argument("--reset", action="store_true", help="Clear progress and re-extract")
    parser.add_argument("--papers", default=str(PAPERS_PATH))
    args = parser.parse_args()

    papers_path = Path(args.papers)
    if not papers_path.exists():
        console.print(f"[red]Papers file not found: {papers_path}[/red]")
        console.print("[yellow]Run first: uv run python scripts/ingest_papers.py[/yellow]")
        sys.exit(1)

    if args.reset:
        for p in (ENTITIES_PATH, EXTRACTION_PROGRESS_PATH, CANONICAL_IDS_PATH):
            if p.exists():
                p.unlink()
                console.print(f"[yellow]Deleted {p}[/yellow]")

    if args.max:
        all_lines = papers_path.read_text().strip().splitlines()
        sliced = Path("/tmp/papers_slice.jsonl")
        sliced.write_text("\n".join(all_lines[: args.max]))
        papers_path = sliced
        console.print(f"[dim]Testing with {args.max} papers[/dim]")

    console.print(f"[cyan]Starting entity extraction from {papers_path}...[/cyan]")
    console.print("[dim]Rate-limited to ~1 batch/second. Cost: ~$0.03 per 10 papers.[/dim]\n")

    client = anthropic.Anthropic()
    results = extract_all(papers_path=papers_path, client=client)

    total_entities = sum(len(r.entities) for r in results)
    total_rels = sum(len(r.relationships) for r in results)

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Papers processed this run: [bold]{len(results)}[/bold]")
    console.print(f"  Total entities extracted:  [bold]{total_entities}[/bold]")
    console.print(f"  Total relationships:        [bold]{total_rels}[/bold]")
    console.print(f"  Output: {ENTITIES_PATH}")

    if CANONICAL_IDS_PATH.exists():
        registry = json.loads(CANONICAL_IDS_PATH.read_text())
        console.print(f"  Canonical IDs: {len(registry)}")

    console.print("\nNext step:")
    console.print("  [dim]uv run python scripts/build_graph.py[/dim]")


if __name__ == "__main__":
    main()
