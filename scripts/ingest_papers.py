#!/usr/bin/env python3
"""
Ingest ALS papers from PubMed + PMC full text + Semantic Scholar citation counts.

Usage:
    uv run python scripts/ingest_papers.py                         # default query, 500 papers
    uv run python scripts/ingest_papers.py --max 10                # small test run
    uv run python scripts/ingest_papers.py --pmid-file pmids.txt   # from curated PMID list
    uv run python scripts/ingest_papers.py --skip-fulltext         # abstracts only
    uv run python scripts/ingest_papers.py --skip-citations        # skip Semantic Scholar
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.progress import track

import ingestion.pmc as pmc
import ingestion.pubmed as pubmed
import ingestion.semantic_scholar as ss
from config import PAPERS_PATH, PUBMED_DEFAULT_MAX, PUBMED_DEFAULT_QUERY, PUBMED_REFRESH_QUERY_TEMPLATE

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ALS papers")
    parser.add_argument("--query", default=None, help="PubMed query string (overrides --since)")
    parser.add_argument("--since", metavar="YYYY-MM-DD", help="Fetch papers published on or after this date (incremental append mode)")
    parser.add_argument("--max", type=int, default=PUBMED_DEFAULT_MAX, dest="max_results", help="Max papers to fetch")
    parser.add_argument("--pmid-file", help="Path to file with one PMID per line")
    parser.add_argument("--skip-fulltext", action="store_true", help="Skip PMC full text fetch")
    parser.add_argument("--skip-citations", action="store_true", help="Skip Semantic Scholar citation counts")
    args = parser.parse_args()

    if args.query is None:
        if args.since:
            args.query = PUBMED_REFRESH_QUERY_TEMPLATE.format(since_date=args.since)
        else:
            args.query = PUBMED_DEFAULT_QUERY

    PAPERS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Obtain PMIDs
    if args.pmid_file:
        lines = Path(args.pmid_file).read_text().splitlines()
        pmids = [line.strip() for line in lines if line.strip()]
        console.print(f"[cyan]Loaded {len(pmids)} PMIDs from {args.pmid_file}[/cyan]")
    else:
        console.print(f"[cyan]Searching PubMed (max {args.max_results})...[/cyan]")
        pmids = pubmed.search_pmids(args.query, max_results=args.max_results)
        console.print(f"[green]Found {len(pmids)} PMIDs[/green]")

    if not pmids:
        console.print("[red]No PMIDs found — check your query or PMID file.[/red]")
        sys.exit(1)

    # In append mode, filter out PMIDs already in the existing corpus
    existing_pmids: set[str] = set()
    if args.since and PAPERS_PATH.exists():
        with open(PAPERS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_pmids.add(json.loads(line).get("pmid", ""))
        original_count = len(pmids)
        pmids = [p for p in pmids if p not in existing_pmids]
        console.print(f"[cyan]Incremental mode: {original_count} fetched, {len(existing_pmids)} already known, {len(pmids)} new[/cyan]")
        if not pmids:
            console.print("[green]No new papers found — corpus is up to date.[/green]")
            return

    # Step 2: Fetch paper records from PubMed
    console.print("[cyan]Fetching Medline records...[/cyan]")
    papers = pubmed.fetch_by_pmids(pmids)
    console.print(f"[green]Parsed {len(papers)} papers with abstracts[/green]")

    if not papers:
        console.print("[red]No papers with abstracts retrieved.[/red]")
        sys.exit(1)

    # Step 3: Enrich with PMC full text
    if not args.skip_fulltext:
        all_pmids = [p.pmid for p in papers]
        pmcid_map = pmc.get_pmcids(all_pmids)  # shows its own progress bar
        console.print(f"[green]{len(pmcid_map)} papers have PMC full text available[/green]")

        pmid_to_paper = {p.pmid: p for p in papers}
        ft_count = 0
        for pmid, pmcid in track(pmcid_map.items(), description="Fetching full text...", console=console):
            if pmid in pmid_to_paper:
                full_text = pmc.fetch_full_text(pmcid)
                if full_text:
                    pmid_to_paper[pmid].full_text = full_text
                    ft_count += 1
        console.print(f"[green]Retrieved section-parsed full text for {ft_count} papers[/green]")

    # Step 4: Enrich with citation counts from Semantic Scholar
    if not args.skip_citations:
        console.print("[cyan]Fetching citation counts from Semantic Scholar...[/cyan]")
        citation_map = ss.fetch_citation_counts([p.pmid for p in papers])
        for paper in papers:
            paper.citation_count = citation_map.get(paper.pmid, 0)
        console.print(f"[green]Got citation counts for {len(citation_map)}/{len(papers)} papers[/green]")

    # Step 5: Write to JSONL (append in incremental mode, overwrite otherwise)
    write_mode = "a" if args.since else "w"
    with open(PAPERS_PATH, write_mode, encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper.to_dict()) + "\n")

    has_fulltext = sum(1 for p in papers if p.full_text)
    has_citations = sum(1 for p in papers if p.citation_count > 0)

    console.print(f"\n[bold green]Done![/bold green] Written to {PAPERS_PATH}")
    console.print(f"  Papers:         {len(papers)}")
    console.print(f"  With full text: {has_fulltext} ({has_fulltext * 100 // len(papers)}%)")
    console.print(f"  With citations: {has_citations} ({has_citations * 100 // len(papers)}%)")


if __name__ == "__main__":
    main()
