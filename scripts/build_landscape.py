#!/usr/bin/env python3
"""
Build the Experimental ALS Therapy Landscape (offline) — v2: multi-label, grounded, abstaining.

Derives experimental therapies from clinical-trial interventions, then for each therapy:
  1. retrieves genuine mechanism-of-action evidence (RAG over the paper corpus by drug name +
     the therapy's own trial summaries) — NOT the noisy aggregated trial target_entities;
  2. classifies MULTI-LABEL with Claude (Opus 4.8) via the Batch API — every mechanism must be
     backed by a verbatim evidence quote, and the model abstains (empty list) when unsure;
  3. cross-checks each asserted mechanism with a model-independent BioLORD cosine between the
     therapy's evidence and the mechanism-class description (drops weakly-supported labels);
  4. keeps only mechanisms above the confidence threshold τ; a therapy with none is "unclassified"
     (shown honestly as "Mechanism not established"), never guessed.

Writes data/landscape/landscape.json (committed to git; the app loads it at startup).

Usage:
    uv run python scripts/build_landscape.py
    uv run python scripts/build_landscape.py --limit 30   # cheap validation on top 30 therapies
    uv run python scripts/build_landscape.py --reset      # ignore any in-flight batch state
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import anthropic
import numpy as np
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from rich.console import Console

from config import (
    LANDSCAPE_BATCH_STATE_PATH,
    LANDSCAPE_EVIDENCE_ABSTRACTS,
    LANDSCAPE_MIN_CONFIDENCE,
    LANDSCAPE_MODEL,
    LANDSCAPE_PATH,
    LANDSCAPE_XCHECK_MIN_COSINE,
    THERAPY_CLASSES_PATH,
    TRIALS_PATH,
)
from prompts import LANDSCAPE_SYSTEM
from tools import LANDSCAPE_TOOLS

console = Console()

_THERAPEUTIC_TYPES = {"DRUG", "BIOLOGICAL", "DIETARY_SUPPLEMENT", "GENETIC", "COMBINATION_PRODUCT"}
_EXCLUDE_RE = re.compile(
    r"(placebo|sham|matching|best supportive care|standard of care|blood sample|"
    r"saline|vehicle|diagnostic|questionnaire|no intervention|usual care|dextrose)",
    re.IGNORECASE,
)
_MODALITY_PREFIX_RE = re.compile(
    r"^(drug|biological|device|other|dietary supplement|genetic|procedure|"
    r"combination product|radiation|behavioral|diagnostic test)\s*:\s*",
    re.IGNORECASE,
)
_STATUS_GROUP = {
    "RECRUITING": "recruiting", "NOT_YET_RECRUITING": "recruiting", "ENROLLING_BY_INVITATION": "recruiting",
    "ACTIVE_NOT_RECRUITING": "active",
    "COMPLETED": "completed", "APPROVED_FOR_MARKETING": "completed", "AVAILABLE": "completed",
    "TERMINATED": "terminated", "WITHDRAWN": "terminated", "SUSPENDED": "terminated",
    "NO_LONGER_AVAILABLE": "terminated", "TEMPORARILY_NOT_AVAILABLE": "terminated",
}
_GROUP_ORDER = {"recruiting": 0, "active": 1, "completed": 2, "terminated": 3, "other": 4}
_THERAPIES_PER_REQUEST = 10   # richer evidence per therapy → smaller batches
_POLL_INTERVAL_S = 30


def _norm_name(name: str) -> str:
    name = _MODALITY_PREFIX_RE.sub("", name or "").strip()
    return re.sub(r"\s+", " ", name)


def _status_group(status: str) -> str:
    return _STATUS_GROUP.get((status or "").upper(), "other")


def _load_trials(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def group_therapies(trials: list[dict]) -> dict[str, dict]:
    """Group trials by normalized therapeutic intervention name."""
    groups: dict[str, dict] = {}
    for t in trials:
        trial_meta = {
            "nct_id": t.get("nct_id", ""), "title": t.get("title", ""), "phase": t.get("phase", ""),
            "status": t.get("status", ""), "status_group": _status_group(t.get("status", "")),
            "start_date": t.get("start_date", ""), "sponsor": t.get("sponsor", ""),
            "url": t.get("url", "") or f"https://clinicaltrials.gov/study/{t.get('nct_id','')}",
            "summary": t.get("summary", ""),
        }
        for iv in t.get("interventions", []):
            if iv.get("type") not in _THERAPEUTIC_TYPES:
                continue
            name = _norm_name(iv.get("name", ""))
            if not name or _EXCLUDE_RE.search(name):
                continue
            key = name.lower()
            g = groups.setdefault(key, {"display": name, "raw_names": set(), "trials": {}})
            g["raw_names"].add(iv.get("name", ""))
            g["trials"][trial_meta["nct_id"]] = trial_meta
    for g in groups.values():
        g["trials"] = list(g["trials"].values())
    return groups


# ── evidence retrieval + cross-check embeddings ──────────────────────────────

def _load_collection():
    try:
        from config import CHROMA_COLLECTION, CHROMA_DIR
        from rag.indexer import load_collection
        return load_collection(CHROMA_DIR, CHROMA_COLLECTION)
    except Exception as e:
        console.print(f"[yellow]No chroma collection ({e}); evidence = trial summaries only, no cross-check[/yellow]")
        return None


def _embed(texts: list[str]) -> np.ndarray:
    from rag.indexer import _EMBED_FN
    vecs = np.asarray(_EMBED_FN(texts), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-9, None)


def _retrieve_evidence(collection, g: dict) -> str:
    """MoA evidence for a therapy: top abstracts (by drug name) + its own trial summaries."""
    parts: list[str] = []
    summaries = [t["summary"] for t in g["trials"][:3] if t.get("summary")]
    parts.extend(s[:400] for s in summaries)
    if collection is not None:
        try:
            from rag.retriever import search
            for r in search(collection, g["display"], n_results=LANDSCAPE_EVIDENCE_ABSTRACTS):
                doc = (r.get("document") or "")[:500]
                if doc:
                    parts.append(doc)
        except Exception:
            pass
    return "\n".join(parts)[:3000]


# ── LLM batch ────────────────────────────────────────────────────────────────

def _therapy_context(key: str, g: dict) -> str:
    lines = [
        f"--- THERAPY_KEY:{key} ---",
        f"Intervention name(s): {', '.join(sorted(g['raw_names']))[:160]}",
        "EVIDENCE (paper abstracts about this therapy + its trial summaries):",
        g.get("evidence", "(no evidence retrieved)"),
    ]
    return "\n".join(lines)


def _format_batch(batch: list[tuple[str, dict]]) -> str:
    parts = [
        f"Classify each of the following {len(batch)} ALS experimental therapies from its EVIDENCE. "
        "Call classify_therapy once per therapy (echo therapy_key). Emit only mechanisms you can quote "
        "from the evidence; return an empty mechanisms list if the evidence establishes none.\n"
    ]
    parts.extend(_therapy_context(key, g) for key, g in batch)
    return "\n\n".join(parts)


def _build_params(batch: list[tuple[str, dict]]) -> MessageCreateParamsNonStreaming:
    return MessageCreateParamsNonStreaming(
        model=LANDSCAPE_MODEL,
        max_tokens=8192,
        system=LANDSCAPE_SYSTEM,
        tools=list(LANDSCAPE_TOOLS),
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": _format_batch(batch)}],
    )


def _run_batch(client, custom_id_to_batch: dict, state_path: Path, reset: bool) -> dict[str, dict]:
    batch = None
    if not reset and state_path.exists():
        try:
            existing = client.messages.batches.retrieve(json.loads(state_path.read_text())["batch_id"])
            if existing.processing_status in {"in_progress", "validating", "finalizing", "ended"}:
                console.print(f"[dim]Resuming batch {existing.id}[/dim]"); batch = existing
        except anthropic.NotFoundError:
            pass
    if batch is None:
        requests = [Request(custom_id=cid, params=_build_params(b)) for cid, b in custom_id_to_batch.items()]
        batch = client.messages.batches.create(requests=requests)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"batch_id": batch.id}))
        console.print(f"[cyan]Submitted batch {batch.id} ({len(requests)} requests, model={LANDSCAPE_MODEL})[/cyan]")
    with console.status("Classifying therapies (batch)…"):
        while batch.processing_status != "ended":
            if batch.processing_status in {"canceling", "canceled", "expired"}:
                console.print(f"[red]Batch ended early: {batch.processing_status}[/red]"); break
            time.sleep(_POLL_INTERVAL_S)
            batch = client.messages.batches.retrieve(batch.id)
    records: dict[str, dict] = {}
    for res in client.messages.batches.results(batch.id):
        if res.result.type != "succeeded":
            console.print(f"[yellow]Request {res.custom_id} {res.result.type}[/yellow]"); continue
        for block in res.result.message.content:
            if block.type == "tool_use" and block.name == "classify_therapy":
                key = str(block.input.get("therapy_key", "")).strip()
                if key:
                    records[key] = block.input
    state_path.unlink(missing_ok=True)
    return records


# ── filtering: confidence threshold + BioLORD cross-check ────────────────────

def _filter_all(records: dict, class_emb: dict, valid: set) -> dict[str, list[dict]]:
    """Per-mechanism filter: confidence ≥ τ AND justification↔class BioLORD cosine ≥ threshold.

    The cross-check embeds each mechanism's own justification (evidence_quote) and compares it to
    the claimed class description — catching internally-inconsistent labels (justification is about
    a different mechanism than the class claimed). Batched so all embeddings are one pass.
    """
    cand = []  # (key, class, role, conf, quote)
    for key, rec in records.items():
        for m in rec.get("mechanisms", []) or []:
            cls = m.get("class")
            if cls not in valid or float(m.get("confidence", 0.0)) < LANDSCAPE_MIN_CONFIDENCE:
                continue
            cand.append((key, cls, m.get("role", "contributing"),
                         float(m.get("confidence", 0.0)), (m.get("evidence_quote") or "")[:300]))

    cosines = [None] * len(cand)
    if class_emb and cand:
        embs = _embed([q or cls for (_, cls, _, _, q) in cand])
        for i, (_, cls, _, _, _) in enumerate(cand):
            if cls in class_emb:
                cosines[i] = float(np.dot(embs[i], class_emb[cls]))

    out: dict[str, list[dict]] = {}
    for i, (key, cls, role, conf, quote) in enumerate(cand):
        cos = cosines[i]
        if cos is not None and cos < LANDSCAPE_XCHECK_MIN_COSINE:
            continue  # justification doesn't semantically match the claimed class
        out.setdefault(key, []).append({
            "class": cls, "role": role, "confidence": round(conf, 2),
            "xcheck_cosine": round(cos, 3) if cos is not None else None,
            "evidence": quote,
        })
    for lst in out.values():
        lst.sort(key=lambda m: (m["role"] != "primary", -m["confidence"]))
    return out


# ── assembly ─────────────────────────────────────────────────────────────────

def _trial_block(m: dict) -> tuple[list[dict], dict]:
    trials = sorted(m["_trials"].values(), key=lambda t: (_GROUP_ORDER.get(t["status_group"], 9), t.get("start_date", "")))
    for t in trials:
        t.pop("summary", None)
    counts = {g: 0 for g in _GROUP_ORDER}
    for t in trials:
        counts[t["status_group"]] += 1
    counts["total"] = len(trials)
    return trials, counts


def build_landscape(limit=None, reset=False) -> dict:
    taxonomy = json.loads(THERAPY_CLASSES_PATH.read_text())["classes"]
    valid = {c["name"] for c in taxonomy}
    trials = _load_trials(TRIALS_PATH)
    groups = group_therapies(trials)
    console.print(f"[cyan]{len(trials)} trials → {len(groups)} candidate therapies[/cyan]")

    items = sorted(groups.items(), key=lambda kv: -len(kv[1]["trials"]))
    if limit:
        items = items[:limit]
        console.print(f"[dim]--limit {limit}: top {len(items)} therapies[/dim]")

    # evidence retrieval + cross-check embeddings (local, no API cost)
    collection = _load_collection()
    class_emb = {}
    if collection is not None:
        console.print("[dim]Embedding class descriptions + retrieving evidence (BioLORD)…[/dim]")
        cls_vecs = _embed([f"{c['name']}: {c['description']}" for c in taxonomy])
        class_emb = {c["name"]: cls_vecs[i] for i, c in enumerate(taxonomy)}
    for _, g in items:
        g["evidence"] = _retrieve_evidence(collection, g)

    # classify
    batches = [items[i : i + _THERAPIES_PER_REQUEST] for i in range(0, len(items), _THERAPIES_PER_REQUEST)]
    client = anthropic.Anthropic()
    records = _run_batch(client, {f"batch-{i}": b for i, b in enumerate(batches)}, LANDSCAPE_BATCH_STATE_PATH, reset)
    console.print(f"[green]Classified {len(records)}/{len(items)} therapies[/green]")

    # confidence threshold + justification↔class cross-check (batched)
    filtered = _filter_all(records, class_emb, valid)

    # merge by canonical name, attach trials
    merged: dict[str, dict] = {}
    for key, g in items:
        rec = records.get(key)
        if not rec:
            continue
        mechs = filtered.get(key, [])
        canonical = (rec.get("canonical_name") or g["display"]).strip()
        m = merged.setdefault(canonical.lower(), {
            "name": canonical, "modality": rec.get("modality", "Other"),
            "target": rec.get("target", "Unknown"), "aliases": set(),
            "_mechs": {}, "_trials": {},
        })
        m["aliases"].update(a for a in rec.get("aliases", []) if a)
        m["aliases"].update(g["raw_names"])
        for me in mechs:  # highest-confidence per class wins
            cur = m["_mechs"].get(me["class"])
            if cur is None or me["confidence"] > cur["confidence"]:
                m["_mechs"][me["class"]] = me
        for t in g["trials"]:
            m["_trials"][t["nct_id"]] = t

    # finalize therapies
    therapies, abstained = [], []
    for m in merged.values():
        trials_, counts = _trial_block(m)
        mechs = sorted(m["_mechs"].values(), key=lambda x: (x["role"] != "primary", -x["confidence"]))
        base = {
            "name": m["name"], "modality": m["modality"], "target": m["target"],
            "aliases": sorted(a for a in m["aliases"] if a and a.lower() != m["name"].lower())[:8],
            "mechanisms": mechs, "trials": trials_, "trial_counts": counts,
        }
        (therapies if mechs else abstained).append(base)

    # group into taxonomy buckets (a therapy appears under EACH of its mechanism classes)
    by_class: dict[str, list[dict]] = {}
    for th in therapies:
        for me in th["mechanisms"]:
            entry = {**th, "role_here": me["role"], "confidence_here": me["confidence"]}
            by_class.setdefault(me["class"], []).append(entry)

    classifications = []
    for cls in taxonomy:
        ths = by_class.get(cls["name"], [])
        if not ths:
            continue
        ths.sort(key=lambda t: (t["role_here"] != "primary", -t["trial_counts"]["recruiting"], -t["trial_counts"]["total"]))
        classifications.append({
            "id": cls["id"], "name": cls["name"], "description": cls["description"],
            "primary_count": sum(1 for t in ths if t["role_here"] == "primary"),
            "therapy_count": len(ths),
            "trial_count": sum(t["trial_counts"]["total"] for t in ths if t["role_here"] == "primary"),
            "therapies": ths,
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": LANDSCAPE_MODEL,
        "thresholds": {"min_confidence": LANDSCAPE_MIN_CONFIDENCE, "min_xcheck_cosine": LANDSCAPE_XCHECK_MIN_COSINE},
        "source": {"trials": len(trials), "therapies": len(therapies), "abstained": len(abstained)},
        "classifications": classifications,
        "unclassified": sorted(abstained, key=lambda t: -t["trial_counts"]["total"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ALS experimental therapy landscape (v2)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    if not TRIALS_PATH.exists():
        console.print(f"[red]Trials file not found: {TRIALS_PATH}[/red]"); sys.exit(1)

    landscape = build_landscape(limit=args.limit, reset=args.reset)
    LANDSCAPE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LANDSCAPE_PATH.write_text(json.dumps(landscape, indent=2))

    s = landscape["source"]
    console.print(f"\n[bold green]Done![/bold green] → {LANDSCAPE_PATH}")
    console.print(f"  Therapies classified: [bold]{s['therapies']}[/bold] · abstained (insufficient evidence): [bold]{s['abstained']}[/bold]")
    for c in landscape["classifications"]:
        console.print(f"  [dim]{c['name']}: {c['primary_count']} primary / {c['therapy_count']} incl. contributing[/dim]")


if __name__ == "__main__":
    main()
