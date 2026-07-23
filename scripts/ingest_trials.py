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

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ALS clinical trials")
    parser.add_argument("--upsert", action="store_true", help="Merge fetched trials into existing trials.jsonl by nct_id")
    args = parser.parse_args()

    TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()

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

    with open(TRIALS_PATH, "w", encoding="utf-8") as f:
        for trial in trials:
            f.write(json.dumps(trial) + "\n")

    with_targets = sum(1 for t in trials if t.get("target_entities"))
    console.print(f"\n[bold green]Done![/bold green] Written to {TRIALS_PATH}")
    console.print(f"  Trials:               {len(trials)}")
    console.print(f"  With entity targets:  {with_targets}")


if __name__ == "__main__":
    main()
