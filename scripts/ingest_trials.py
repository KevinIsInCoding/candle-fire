#!/usr/bin/env python3
"""
Ingest ALS clinical trials from ClinicalTrials.gov v2 API.
Fetches ALL statuses (recruiting, completed, terminated, withdrawn, etc.)
so physicians can see the full trial landscape including failed trials.

Usage:
    uv run python scripts/ingest_trials.py
    uv run python scripts/ingest_trials.py --upsert
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import anthropic
from rich.console import Console

from config import TRIALS_PATH
from ingestion.clinicaltrials import fetch_als_trials

console = Console()

def _write_trials(trials: list[dict]) -> None:
    with open(TRIALS_PATH, "w", encoding="utf-8") as f:
        for trial in trials:
            f.write(json.dumps(trial) + "\n")


def _run_summaries(client: anthropic.Anthropic) -> None:
    """Pipeline step 5.5: attach RAG-grounded mechanism summaries to the ingested trials.

    Requires the ChromaDB index (built in step 5) — the animal-results and repurposed-from
    claims are grounded in that corpus. Resumable: re-running only processes trials that lack a
    mechanism_summary, and progress is checkpointed to trials.jsonl after every batch.
    """
    import chromadb

    from config import CHROMA_COLLECTION, CHROMA_DIR
    from ingestion.clinicaltrials import enrich_trial_mechanisms

    if not TRIALS_PATH.exists():
        console.print("[red]No trials.jsonl — run ingest first.[/red]")
        sys.exit(1)
    if not CHROMA_DIR.exists():
        console.print("[red]No ChromaDB index — run scripts/build_index.py (step 5) first.[/red]")
        sys.exit(1)

    with open(TRIALS_PATH, encoding="utf-8") as f:
        trials = [json.loads(line) for line in f if line.strip()]

    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(CHROMA_COLLECTION)
    console.print(f"[cyan]Summarizing mechanisms for {len(trials)} trials (grounded in "
                  f"{collection.count()} chunks)...[/cyan]")

    enrich_trial_mechanisms(trials, collection, client, checkpoint=_write_trials)
    _write_trials(trials)

    with_summary = sum(1 for t in trials if t.get("mechanism_summary"))
    grounded = sum(
        1 for t in trials
        if (t.get("mechanism_summary") or {}).get("animal_results", "unknown") != "unknown"
    )
    console.print(f"\n[bold green]Done![/bold green] Written to {TRIALS_PATH}")
    console.print(f"  With mechanism summary:   {with_summary}")
    console.print(f"  With grounded animal data: {grounded}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ALS clinical trials")
    parser.add_argument("--upsert", action="store_true", help="Merge fetched trials into existing trials.jsonl by nct_id")
    parser.add_argument("--summaries", action="store_true", help="Step 5.5: add RAG-grounded mechanism summaries to existing trials (requires ChromaDB index); does not re-fetch")
    args = parser.parse_args()

    TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()

    if args.summaries:
        _run_summaries(client)
        return

    console.print("[cyan]Fetching all ALS interventional trials (no status filter)...[/cyan]")
    trials = fetch_als_trials(client=client)
    console.print(f"[green]Fetched {len(trials)} trials[/green]")

    if args.upsert and TRIALS_PATH.exists():
        existing: dict[str, dict] = {}
        with open(TRIALS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    t = json.loads(line)
                    existing[t["nct_id"]] = t
        before = len(existing)
        for trial in trials:
            existing[trial["nct_id"]] = trial
        trials = list(existing.values())
        console.print(f"[cyan]Upsert: {before} existing + {len(trials) - before} new/updated → {len(trials)} total[/cyan]")

    _write_trials(trials)

    with_targets = sum(1 for t in trials if t.get("target_entities"))
    console.print(f"\n[bold green]Done![/bold green] Written to {TRIALS_PATH}")
    console.print(f"  Trials:               {len(trials)}")
    console.print(f"  With entity targets:  {with_targets}")


if __name__ == "__main__":
    main()
