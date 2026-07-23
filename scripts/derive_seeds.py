#!/usr/bin/env python3
"""
Derive dynamic seed entities from the ingested trial and paper corpus.

Sources:
  1. trials.jsonl — DRUG/BIOLOGICAL interventions + LLM-extracted targets
  2. entities.jsonl — entities appearing in >= SEED_PROMOTION_THRESHOLD papers

Writes data/seeds/derived_seeds.json (always overwritten, never hand-edited).
Run after extract_entities.py and before build_graph.py.

Usage:
    uv run python scripts/derive_seeds.py
    uv run python scripts/derive_seeds.py --threshold 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console

from config import (
    DERIVED_SEEDS_PATH,
    ENTITIES_PATH,
    SEED_PROMOTION_THRESHOLD,
    TRIALS_PATH,
)
from extraction.normalizer import normalize_entity, _GENE_ALIASES, _COMPOUND_ALIASES

console = Console()

_INTERVENTION_BLOCKLIST = {
    "placebo", "standard_care", "standard_of_care", "exercise",
    "physical_therapy", "physiotherapy", "occupational_therapy",
    "riluzole",  # already in manual seeds
    "edaravone",
    "best_supportive_care", "nutritional_support", "sham",
    "observation", "usual_care",
}

_EMPTY_SEEDS: dict[str, list[str]] = {
    "genes": [], "proteins": [], "compounds": [], "mechanisms": [], "phenotypes": [],
}

_TYPE_TO_CATEGORY = {
    "Gene": "genes",
    "Protein": "proteins",
    "Compound": "compounds",
    "Mechanism": "mechanisms",
    "Phenotype": "phenotypes",
    "Pathway": "mechanisms",
}


def _derive_from_trials(trials_path: Path) -> dict[str, set[str]]:
    """Return category → set of display names derived from active trials."""
    result: dict[str, set[str]] = defaultdict(set)

    if not trials_path.exists():
        console.print(f"[yellow]trials.jsonl not found at {trials_path} — skipping trial seeds[/yellow]")
        return result

    n_trials = 0
    with open(trials_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            trial = json.loads(line)
            n_trials += 1

            # Drug/biological interventions from ClinicalTrials.gov structured data
            for iv in trial.get("interventions", []):
                if iv.get("type", "").upper() not in ("DRUG", "BIOLOGICAL"):
                    continue
                name = iv.get("name", "").strip()
                if not name:
                    continue
                canonical = normalize_entity(name, "Compound")
                slug = canonical.split(":", 1)[-1]
                if slug not in _INTERVENTION_BLOCKLIST:
                    result["compounds"].add(name)

            # LLM-extracted target entities
            for target in trial.get("target_entities", []):
                target = target.strip()
                if not target:
                    continue
                # Infer type via alias tables
                u = target.upper()
                if u in _GENE_ALIASES or target in _GENE_ALIASES:
                    result["genes"].add(target)
                elif target in _COMPOUND_ALIASES:
                    result["compounds"].add(target)
                else:
                    # Unknown — put in compounds (most unmatched trial targets are drugs)
                    slug = normalize_entity(target, "Compound").split(":", 1)[-1]
                    if slug not in _INTERVENTION_BLOCKLIST:
                        result["compounds"].add(target)

    console.print(f"  Scanned {n_trials} trials")
    return result


def _derive_from_entities(entities_path: Path, threshold: int) -> dict[str, set[str]]:
    """Return category → set of display names for entities in >= threshold papers."""
    if not entities_path.exists():
        console.print(f"[yellow]entities.jsonl not found at {entities_path} — skipping paper seeds[/yellow]")
        return defaultdict(set)

    # canonical_id → {type, display_name, count}
    counts: dict[str, dict] = {}

    with open(entities_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            for ent in raw.get("entities", []):
                if ent.get("confidence", 0.0) < 0.6:
                    continue
                cid = ent.get("canonical_id", "")
                if not cid:
                    continue
                if cid not in counts:
                    counts[cid] = {
                        "type": ent.get("type", "Unknown"),
                        "display_name": ent.get("name", cid.split(":", 1)[-1]),
                        "count": 0,
                    }
                counts[cid]["count"] += 1

    result: dict[str, set[str]] = defaultdict(set)
    promoted = 0
    for cid, data in counts.items():
        if data["count"] >= threshold:
            category = _TYPE_TO_CATEGORY.get(data["type"])
            if category:
                result[category].add(data["display_name"])
                promoted += 1

    console.print(f"  {len(counts)} unique entities; {promoted} promoted above threshold={threshold}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive seed entities from corpus")
    parser.add_argument("--trials", type=Path, default=TRIALS_PATH)
    parser.add_argument("--entities", type=Path, default=ENTITIES_PATH)
    parser.add_argument("--threshold", type=int, default=SEED_PROMOTION_THRESHOLD)
    parser.add_argument("--output", type=Path, default=DERIVED_SEEDS_PATH)
    args = parser.parse_args()

    console.print("[cyan]Deriving seeds from trials...[/cyan]")
    trial_seeds = _derive_from_trials(args.trials)

    console.print("[cyan]Deriving seeds from paper entities...[/cyan]")
    paper_seeds = _derive_from_entities(args.entities, args.threshold)

    merged: dict[str, list[str]] = {}
    for category in _EMPTY_SEEDS:
        combined = trial_seeds.get(category, set()) | paper_seeds.get(category, set())
        merged[category] = sorted(combined)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, sort_keys=True))

    total = sum(len(v) for v in merged.values())
    console.print(f"\n[bold green]Done![/bold green] Written to {args.output}")
    console.print(f"  Total derived seeds: {total}")
    for cat, names in merged.items():
        if names:
            console.print(f"  {cat}: {len(names)} ({', '.join(names[:5])}{'...' if len(names) > 5 else ''})")


if __name__ == "__main__":
    main()
