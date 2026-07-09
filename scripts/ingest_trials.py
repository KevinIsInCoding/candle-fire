#!/usr/bin/env python3
"""
Ingest ALS clinical trials from ClinicalTrials.gov v2 API.

Usage:
    uv run python scripts/ingest_trials.py
    uv run python scripts/ingest_trials.py --status RECRUITING
    uv run python scripts/ingest_trials.py --status RECRUITING NOT_YET_RECRUITING COMPLETED
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

_ALL_STATUSES = ["RECRUITING", "COMPLETED", "ACTIVE_NOT_RECRUITING", "NOT_YET_RECRUITING"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest ALS clinical trials")
    parser.add_argument(
        "--status",
        nargs="+",
        default=["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING"],
        choices=_ALL_STATUSES,
        metavar="STATUS",
        help=(
            f"One or more trial statuses to fetch (default: RECRUITING NOT_YET_RECRUITING "
            f"ACTIVE_NOT_RECRUITING). Choices: {_ALL_STATUSES}"
        ),
    )
    args = parser.parse_args()

    TRIALS_PATH.parent.mkdir(parents=True, exist_ok=True)

    client = anthropic.Anthropic()

    console.print(f"[cyan]Fetching ALS interventional trials (status={args.status})...[/cyan]")
    trials = fetch_als_trials(status=args.status, client=client)
    console.print(f"[green]Fetched {len(trials)} trials[/green]")

    with open(TRIALS_PATH, "w", encoding="utf-8") as f:
        for trial in trials:
            f.write(json.dumps(trial) + "\n")

    with_targets = sum(1 for t in trials if t.get("target_entities"))
    console.print(f"\n[bold green]Done![/bold green] Written to {TRIALS_PATH}")
    console.print(f"  Trials:               {len(trials)}")
    console.print(f"  With entity targets:  {with_targets}")


if __name__ == "__main__":
    main()
