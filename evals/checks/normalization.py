"""Offline: entity alias-collapse invariant.

Every surface form in a gold group must normalize to the one canonical_id.
This guards the project's core KG invariant — "TDP-43" and "TARDBP" must produce
one node, never two. Pure static tables, no network.
"""
from __future__ import annotations

import json

from extraction.normalizer import normalize_entity
from evals import thresholds as T
from evals.checks.result import CheckResult


def run() -> CheckResult:
    groups = [json.loads(l) for l in T.QUESTIONS_GOLD.parent.joinpath("aliases.jsonl")
              .read_text().splitlines() if l.strip()]

    total = 0
    collapsed = 0
    details: list[str] = []

    for g in groups:
        expected = g["canonical_id"]
        etype = g["entity_type"]
        for surface in g["surface_forms"]:
            total += 1
            got = normalize_entity(surface, etype)
            if got == expected:
                collapsed += 1
            else:
                details.append(f"{g['group']}: {surface!r} → {got!r} (expected {expected!r})")

    accuracy = collapsed / total if total else 0.0
    passed = accuracy >= T.NORMALIZATION_MIN_ACCURACY
    return CheckResult(
        "normalization", "offline", "pass" if passed else "fail",
        f"{collapsed}/{total} surface forms collapse correctly (acc={accuracy:.2f})",
        metrics={"accuracy": round(accuracy, 4), "total": total, "collapsed": collapsed},
        thresholds={"min_accuracy": T.NORMALIZATION_MIN_ACCURACY},
        details=details,
    )
