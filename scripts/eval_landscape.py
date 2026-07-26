#!/usr/bin/env python3
"""
Evaluate the mechanism classifier against the hand-labeled gold set.

Reports multi-label precision / recall / F1 (micro), exact-set-match rate, and abstention rate,
plus a per-therapy breakdown so misclassifications are visible. Physician's rule — misleading is
worse than missing — so we optimize for PRECISION: a wrong label counts against us; an abstention
(no label) does not count as a precision error, only against recall.

Usage:
    uv run python scripts/eval_landscape.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

from config import LANDSCAPE_PATH, THERAPY_GOLD_PATH

console = Console()


def _predicted_index(landscape: dict) -> dict[str, set]:
    """name/alias (lower) -> set of predicted mechanism classes. Abstained therapies map to set()."""
    idx: dict[str, set] = {}
    seen = {}
    def add(th: dict, mech_classes: set):
        for nm in [th["name"], *th.get("aliases", [])]:
            key = nm.lower()
            # a therapy appears in multiple class buckets; union its mechanism set once
            seen.setdefault(key, set()).update(mech_classes)
            idx[key] = seen[key]
    for c in landscape.get("classifications", []):
        for th in c["therapies"]:
            add(th, {m["class"] for m in th.get("mechanisms", [])})
    for th in landscape.get("unclassified", []):
        add(th, set())
    return idx


def main() -> None:
    if not LANDSCAPE_PATH.exists():
        console.print("[red]No landscape.json — run scripts/build_landscape.py first[/red]"); sys.exit(1)
    landscape = json.loads(LANDSCAPE_PATH.read_text())
    gold = json.loads(THERAPY_GOLD_PATH.read_text())["gold"]
    idx = _predicted_index(landscape)

    tp = fp = fn = 0
    exact = matched = abstained = missing = 0
    rows = []
    for g in gold:
        names = [g["therapy"], *g.get("aliases", [])]
        pred = None
        for nm in names:
            if nm.lower() in idx:
                pred = idx[nm.lower()]; break
        gset = set(g["mechanisms"])
        if pred is None:
            missing += 1
            rows.append(("?", g["therapy"], "NOT IN LANDSCAPE", gset, set()))
            continue
        matched += 1
        if not pred:
            abstained += 1
        inter = pred & gset
        tp += len(inter); fp += len(pred - gset); fn += len(gset - pred)
        if pred == gset:
            exact += 1
        mark = "✓" if pred == gset else ("~" if inter else "✗")
        rows.append((mark, g["therapy"], "", gset, pred))

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    console.print("\n[bold]Per-therapy (✓ exact, ~ partial, ✗ wrong):[/bold]")
    for mark, name, note, gset, pred in sorted(rows, key=lambda r: r[0]):
        color = {"✓": "green", "~": "yellow", "✗": "red", "?": "dim"}.get(mark, "white")
        console.print(f"  [{color}]{mark}[/{color}] {name:22s} gold={sorted(gset)} pred={sorted(pred)} {note}")

    console.print(f"\n[bold]Matched {matched}/{len(gold)} gold therapies[/bold] "
                  f"({missing} not in landscape, {abstained} abstained)")
    console.print(f"  Micro precision: [bold]{prec:.2f}[/bold]  recall: {rec:.2f}  F1: {f1:.2f}")
    console.print(f"  Exact-set match: {exact}/{matched}")

    # hard assertions for the flagged cases
    def pred_of(name):
        return idx.get(name.lower())
    console.print("\n[bold]Flagged-case checks:[/bold]")
    cnm = pred_of("CNM-Au8") or set()
    ok_cnm = "Neuroinflammation" not in cnm and ({"Mitochondrial dysfunction", "Oxidative stress"} & cnm)
    console.print(f"  CNM-Au8 not Neuroinflammation & has bioenergetic/oxidative: "
                  f"[{'green' if ok_cnm else 'red'}]{bool(ok_cnm)}[/] (pred={sorted(cnm)})")


if __name__ == "__main__":
    main()
