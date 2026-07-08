#!/usr/bin/env python3
"""
Build the ChromaDB vector index from papers.jsonl.
Run after scripts/ingest_papers.py.

Usage:
    uv run python scripts/build_index.py
    uv run python scripts/build_index.py --reset   # drop and rebuild from scratch
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import argparse

from rich.console import Console
from rich.progress import track

from config import CHROMA_COLLECTION, CHROMA_DIR, PAPERS_PATH
from rag.indexer import build_collection

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB vector index from ALS papers")
    parser.add_argument("--reset", action="store_true", help="Drop existing collection and rebuild")
    parser.add_argument("--papers", default=str(PAPERS_PATH), help="Path to papers.jsonl")
    args = parser.parse_args()

    papers_path = Path(args.papers)
    if not papers_path.exists():
        console.print(f"[red]Papers file not found: {papers_path}[/red]")
        console.print("[yellow]Run first: uv run python scripts/ingest_papers.py[/yellow]")
        sys.exit(1)

    if args.reset:
        console.print("[yellow]Resetting collection...[/yellow]")

    console.print(f"[cyan]Building ChromaDB index from {papers_path}...[/cyan]")
    console.print("[dim]First run downloads the embedding model (~80MB). Subsequent runs are fast.[/dim]\n")

    collection = build_collection(
        papers_path=papers_path,
        chroma_dir=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION,
        reset=args.reset,
    )

    console.print(f"\n[bold green]Done![/bold green] Collection '{CHROMA_COLLECTION}' at {CHROMA_DIR}")
    console.print(f"  Total chunks indexed: [bold]{collection.count()}[/bold]")
    console.print("\nTest a query:")
    console.print("  [dim]uv run python main.py \"What is the evidence for tofersen targeting SOD1?\"[/dim]")


if __name__ == "__main__":
    main()
