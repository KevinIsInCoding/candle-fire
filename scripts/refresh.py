#!/usr/bin/env python3
"""
Incremental refresh orchestrator. Runs the full pipeline in order, using
--since for papers and --upsert for trials to avoid full re-ingestion.

State is tracked in data/.refresh_state.json. On first run (no state file),
defaults to fetching papers since 2024-12-31 (end of the initial corpus window).

Usage:
    uv run python scripts/refresh.py                    # incremental from last state
    uv run python scripts/refresh.py --since 2025-01-01
    uv run python scripts/refresh.py --dry-run
    uv run python scripts/refresh.py --skip-papers      # trials + graph only
    uv run python scripts/refresh.py --full-rebuild     # ignore state, full re-ingest
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console

from config import REFRESH_STATE_PATH

console = Console()

_SCRIPTS_DIR = Path(__file__).parent
_DEFAULT_SINCE = "2024-12-31"


def _load_state() -> dict:
    if REFRESH_STATE_PATH.exists():
        return json.loads(REFRESH_STATE_PATH.read_text())
    return {}


def _save_state(state: dict) -> None:
    REFRESH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REFRESH_STATE_PATH.write_text(json.dumps(state, indent=2))


def _run(script: str, *args: str, dry_run: bool = False) -> None:
    cmd = [sys.executable, str(_SCRIPTS_DIR / script), *args]
    console.print(f"[cyan]{'(dry-run) ' if dry_run else ''}Running:[/cyan] {' '.join(cmd)}")
    if dry_run:
        return
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        console.print(f"[red]Script {script} exited with code {result.returncode}[/red]")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Incremental pipeline refresh")
    parser.add_argument("--since", metavar="YYYY-MM-DD", help="Override since-date for paper ingestion")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without executing")
    parser.add_argument("--skip-papers", action="store_true", help="Skip paper ingestion (trials + graph only)")
    parser.add_argument("--full-rebuild", action="store_true", help="Full re-ingest, ignoring state")
    args = parser.parse_args()

    state = _load_state()
    today = str(date.today())

    if args.full_rebuild:
        console.print("[yellow]Full rebuild requested — ignoring refresh state[/yellow]")
        since_date = None
    elif args.since:
        since_date = args.since
    else:
        since_date = state.get("last_papers_ingest", _DEFAULT_SINCE)

    console.print(f"[bold]Candle-fire incremental refresh[/bold] — {today}")
    if not args.full_rebuild and not args.skip_papers:
        console.print(f"  Fetching papers since: {since_date}")

    # 1. Papers
    if not args.skip_papers:
        if args.full_rebuild or since_date is None:
            _run("ingest_papers.py", "--skip-fulltext", "--skip-citations", dry_run=args.dry_run)
        else:
            _run("ingest_papers.py", "--since", since_date, "--skip-fulltext", "--skip-citations", dry_run=args.dry_run)

    # 2. Trials
    _run("ingest_trials.py", "--upsert", dry_run=args.dry_run)

    # 3. Entity extraction (auto-resumes via .progress.json)
    _run("extract_entities.py", dry_run=args.dry_run)

    # 4. Derive seeds
    _run("derive_seeds.py", dry_run=args.dry_run)

    # 5. Build graph
    _run("build_graph.py", dry_run=args.dry_run)

    # 6. Build index (idempotent — skips existing chunks)
    _run("build_index.py", dry_run=args.dry_run)

    if not args.dry_run:
        new_state = {**state, "last_papers_ingest": today, "last_trials_ingest": today}
        _save_state(new_state)
        console.print(f"\n[bold green]Refresh complete.[/bold green] State written to {REFRESH_STATE_PATH}")
    else:
        console.print("\n[yellow]Dry run complete — no changes made.[/yellow]")


if __name__ == "__main__":
    main()
