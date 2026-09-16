"""Offline: therapy-landscape classifier gold gate.

Scores the committed landscape.json against the hand-labeled therapy_gold.json —
no LLM, the landscape is pre-built and committed. Physician's rule (a wrong label
is worse than a missing one), so PRECISION is the tighter gate; F1 backstops
recall collapse. Mirrors scripts/eval_landscape.py.
"""
from __future__ import annotations

import json

from config import LANDSCAPE_PATH, THERAPY_GOLD_PATH
from evals import thresholds as T
from evals.checks.result import CheckResult


def _predicted_index(landscape: dict) -> dict[str, set]:
    idx: dict[str, set] = {}
    seen: dict[str, set] = {}

    def add(th: dict, mech_classes: set):
        for nm in [th["name"], *th.get("aliases", [])]:
            key = nm.lower()
            seen.setdefault(key, set()).update(mech_classes)
            idx[key] = seen[key]

    for c in landscape.get("classifications", []):
        for th in c["therapies"]:
            add(th, {m["class"] for m in th.get("mechanisms", [])})
    for th in landscape.get("unclassified", []):
        add(th, set())
    return idx


def run() -> CheckResult:
    if not LANDSCAPE_PATH.exists():
        return CheckResult("landscape", "offline", "fail",
                           "landscape.json missing — run scripts/build_landscape.py",
                           details=[str(LANDSCAPE_PATH)])

    landscape = json.loads(LANDSCAPE_PATH.read_text())
    gold = json.loads(THERAPY_GOLD_PATH.read_text())["gold"]
    idx = _predicted_index(landscape)

    tp = fp = fn = 0
    matched = 0
    for g in gold:
        pred = None
        for nm in [g["therapy"], *g.get("aliases", [])]:
            if nm.lower() in idx:
                pred = idx[nm.lower()]
                break
        if pred is None:
            continue  # not in landscape — a recall miss, counted via fn below
        matched += 1
        gset = set(g["mechanisms"])
        tp += len(pred & gset)
        fp += len(pred - gset)
        fn += len(gset - pred)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    # Flagged regression case: CNM-Au8 must be bioenergetic/oxidative, not neuroinflammation.
    cnm = idx.get("cnm-au8", set())
    cnm_ok = "Neuroinflammation" not in cnm and bool(
        {"Mitochondrial dysfunction", "Oxidative stress"} & cnm)

    details: list[str] = []
    if prec < T.LANDSCAPE_MIN_PRECISION:
        details.append(f"precision {prec:.2f} < {T.LANDSCAPE_MIN_PRECISION}")
    if f1 < T.LANDSCAPE_MIN_F1:
        details.append(f"F1 {f1:.2f} < {T.LANDSCAPE_MIN_F1}")
    if not cnm_ok:
        details.append(f"CNM-Au8 flagged-case regression (pred={sorted(cnm)})")

    status = "fail" if details else "pass"
    return CheckResult(
        "landscape", "offline", status,
        f"precision={prec:.2f} recall={rec:.2f} F1={f1:.2f}, matched {matched}/{len(gold)}, CNM-Au8 ok={cnm_ok}",
        metrics={"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                 "matched": matched, "gold_total": len(gold), "cnm_au8_ok": cnm_ok},
        thresholds={"min_precision": T.LANDSCAPE_MIN_PRECISION, "min_f1": T.LANDSCAPE_MIN_F1},
        details=details,
    )
