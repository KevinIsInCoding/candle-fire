"""Tier-1 red/green eval gate.

    uv run python -m evals.runner --suite offline   # PR CI (no data, no LLM)
    uv run python -m evals.runner --suite data      # deploy gate (needs chroma+graph)
    uv run python -m evals.runner --suite all       # offline + data (the red/green gate)
    uv run python -m evals.runner --suite judge     # Tier-2 LLM judge (nightly; needs data + API key)

Exit 0 = green (all non-skipped checks pass). Exit 1 = red (any check failed, or a
data check couldn't find its runtime assets when the data suite was requested).
Writes a JSON report to evals/report/latest.json for trend tracking.

`all` is the deterministic gate (offline + data); the `judge` suite is separate and
never part of `all` — it is non-deterministic and costs tokens, so it runs on its own
nightly schedule and never blocks a PR.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

from evals import thresholds as T
from evals.checks import landscape, normalization, schema  # offline: stdlib-only imports
from evals.checks.result import CheckResult
# Data checks (citations/kg_expansion/retrieval) pull in chromadb + torch via
# rag.retriever, so they're imported lazily inside _run_data() — the offline
# suite (PR CI) then runs on the standard library alone, no ML stack install.

_GREEN, _RED, _YEL, _DIM, _BOLD, _RST = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
_MARK = {"pass": f"{_GREEN}PASS{_RST}", "fail": f"{_RED}FAIL{_RST}", "skip": f"{_DIM}SKIP{_RST}"}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _run_offline() -> list[CheckResult]:
    return [schema.run(), normalization.run(), landscape.run()]


def _run_data() -> list[CheckResult]:
    from evals.checks import citations, kg_expansion, retrieval
    from evals.checks.context import DataContext, data_assets_present, load_questions

    if not data_assets_present():
        # Requested the data suite but the runtime assets aren't here. That's a
        # red condition for a deploy gate — we must not deploy unverified.
        return [CheckResult("data-assets", "data", "fail",
                            "ChromaDB index / graph pickle not found locally — "
                            "run the pipeline or fetch runtime data before the data suite")]
    ctx = DataContext(questions=load_questions())
    return [citations.run(ctx), kg_expansion.run(ctx), retrieval.run(ctx)]


def _run_judge() -> list[CheckResult]:
    from evals.checks import synthesis
    from evals.checks.context import DataContext, data_assets_present, load_questions

    if not data_assets_present():
        return [CheckResult("data-assets", "data", "fail",
                            "ChromaDB index / graph pickle not found locally — "
                            "fetch runtime data before the judge suite")]
    ctx = DataContext(questions=load_questions())
    return synthesis.run(ctx)


def _print(results: list[CheckResult]) -> None:
    print(f"\n{_BOLD}Candle-Fire eval gate{_RST}  (commit {_git_sha()})\n")
    for r in results:
        print(f"  {_MARK[r.status]}  {r.name:16s} {r.summary}")
        for d in r.details:
            print(f"          {_YEL}· {d}{_RST}")
    print()


def _write_report(results: list[CheckResult], suite: str) -> None:
    T.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": _git_sha(),
        "suite": suite,
        "green": not any(r.failed for r in results),
        "checks": [r.to_dict() for r in results],
    }
    (T.REPORT_DIR / "latest.json").write_text(json.dumps(report, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description="Candle-Fire Tier-1 eval gate")
    ap.add_argument("--suite", choices=["offline", "data", "all", "judge"], default="all")
    args = ap.parse_args()

    results: list[CheckResult] = []
    if args.suite in ("offline", "all"):
        results += _run_offline()
    if args.suite in ("data", "all"):
        results += _run_data()
    if args.suite == "judge":
        results += _run_judge()

    _print(results)
    _write_report(results, args.suite)

    failed = [r for r in results if r.failed]
    skipped = [r for r in results if r.skipped]
    if failed:
        print(f"{_RED}{_BOLD}RED — {len(failed)} check(s) failed.{_RST}")
        return 1
    note = f" ({len(skipped)} skipped)" if skipped else ""
    print(f"{_GREEN}{_BOLD}GREEN — all checks passed{note}.{_RST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
