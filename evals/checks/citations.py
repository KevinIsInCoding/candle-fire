"""Data: citation existence gate (zero tolerance).

Every reference/citation PMID named in the gold set must resolve to a paper in
the corpus. A cited PMID the corpus can't produce is the fabrication risk this
whole eval exists to prevent, so any miss fails the build. Deterministic — no LLM.

This is the deterministic half of attribution: it proves the cited paper *exists*
and is retrievable. Whether that paper actually *supports* the claim is the
Tier-2 (judge) attribution-precision metric, out of scope for the red/green gate.
"""
from __future__ import annotations

from rag.retriever import get_paper
from evals import thresholds as T
from evals.checks.result import CheckResult
from evals.checks.context import DataContext


def run(ctx: DataContext) -> CheckResult:
    pmids: set[str] = set()
    for q in ctx.questions:
        pmids.update(str(p) for p in q.get("reference_citations", []))
        pmids.update(str(p) for p in q.get("relevant_pmids", []))

    if not pmids:
        return CheckResult("citations", "data", "skip",
                           "no cited PMIDs in gold set")

    missing = [p for p in sorted(pmids) if get_paper(ctx.collection, p) is None]
    validity = 1 - len(missing) / len(pmids)
    passed = validity >= T.CITATION_MIN_VALIDITY
    return CheckResult(
        "citations", "data", "pass" if passed else "fail",
        f"{len(pmids) - len(missing)}/{len(pmids)} cited PMIDs exist in corpus (validity={validity:.2f})",
        metrics={"validity": round(validity, 4), "total": len(pmids), "missing": len(missing)},
        thresholds={"min_validity": T.CITATION_MIN_VALIDITY},
        details=[f"PMID {p} not in corpus" for p in missing],
    )
