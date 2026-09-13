"""Backfill enrollment/eligibility criteria into an existing trials.jsonl.

Adds the `eligibility` field to trials that don't have it by fetching each trial's
eligibilityModule from ClinicalTrials.gov v2 — WITHOUT re-running the expensive LLM target
extraction that a full `ingest_trials.py` would. Fetches in batches via `filter.ids`, merges
in place, and rewrites trials.jsonl. Idempotent: re-running only fills trials still missing it.

Usage:
  uv run python scripts/backfill_eligibility.py            # backfill missing eligibility
  uv run python scripts/backfill_eligibility.py --force    # refetch for ALL trials
  uv run python scripts/backfill_eligibility.py --dry-run  # report only, no write

After it writes trials.jsonl, deploy per the runbook:
  uv run python scripts/upload_data.py --only trials   # push to the HF dataset repo
  git push hf main
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CTGOV_BASE, TRIALS_PATH  # noqa: E402
from ingestion.clinicaltrials import _extract_eligibility  # noqa: E402

_BATCH = 50          # NCT ids per request (filter.ids)
_PAUSE_S = 0.34      # ~3 req/s, polite to CT.gov


def _fetch_eligibility(nct_ids: list[str]) -> dict[str, dict]:
    """{nct_id: eligibility dict} for a batch of NCT ids."""
    resp = httpx.get(
        CTGOV_BASE,
        params={
            "filter.ids": ",".join(nct_ids),
            "fields": "NCTId,EligibilityModule",
            "pageSize": len(nct_ids),
        },
        timeout=30,
    )
    resp.raise_for_status()
    out: dict[str, dict] = {}
    for study in resp.json().get("studies", []):
        proto = study.get("protocolSection", {})
        nct = proto.get("identificationModule", {}).get("nctId", "")
        if nct:
            out[nct] = _extract_eligibility(proto.get("eligibilityModule", {}))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill eligibility criteria into trials.jsonl.")
    ap.add_argument("--force", action="store_true", help="Refetch for all trials, not just missing.")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change; write nothing.")
    args = ap.parse_args()

    if not TRIALS_PATH.exists():
        print(f"ERROR — {TRIALS_PATH} not found (run ingest_trials.py first).", file=sys.stderr)
        return 1

    trials = [json.loads(line) for line in TRIALS_PATH.read_text().splitlines() if line.strip()]
    todo = [
        t for t in trials
        if t.get("nct_id") and (args.force or not (t.get("eligibility") or {}).get("criteria"))
    ]
    print(f"{len(trials)} trials; {len(todo)} to fetch eligibility for"
          f"{' (force)' if args.force else ''}.")
    if not todo:
        print("Nothing to do.")
        return 0
    if args.dry_run:
        print("--dry-run: no fetch, no write.")
        return 0

    by_nct = {t["nct_id"]: t for t in trials}
    fetched = 0
    for i in range(0, len(todo), _BATCH):
        ids = [t["nct_id"] for t in todo[i : i + _BATCH]]
        try:
            for nct, elig in _fetch_eligibility(ids).items():
                if nct in by_nct:
                    by_nct[nct]["eligibility"] = elig
                    fetched += 1
        except httpx.HTTPError as exc:
            print(f"  batch {i // _BATCH} failed: {exc}", file=sys.stderr)
        print(f"  {min(i + _BATCH, len(todo))}/{len(todo)}")
        if i + _BATCH < len(todo):
            time.sleep(_PAUSE_S)

    with_crit = sum(1 for t in trials if (t.get("eligibility") or {}).get("criteria"))
    TRIALS_PATH.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trials))
    print(f"Fetched {fetched}; {with_crit}/{len(trials)} trials now have eligibility criteria.")
    print(f"Wrote {TRIALS_PATH}. Next: upload_data.py --only trials, then git push hf main.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
