"""Rendering helpers for the Experimental ALS Therapy Landscape tab (v2: multi-label).

Loads data/landscape/landscape.json and turns it into a Plotly sunburst (mechanism class →
primary therapy) plus a detail panel that lists every mechanism a therapy acts through (with role,
confidence, and justification) and a trials table with status badges + NCT links. Therapies whose
mechanism could not be established are shown honestly under "Insufficient evidence".
"""
from __future__ import annotations

import json
import html

from config import LANDSCAPE_PATH

_INSUFFICIENT = "Insufficient evidence"

_CLASS_COLORS = {
    "TDP-43 proteinopathy": "#6C5CE7", "SOD1": "#0984E3", "C9orf72": "#00B894", "FUS": "#00CEC9",
    "Neuroinflammation": "#E17055", "Oxidative stress": "#D63031", "Mitochondrial dysfunction": "#E84393",
    "Glutamate excitotoxicity": "#FDCB6E", "Proteostasis / autophagy": "#A29BFE",
    "RNA metabolism": "#74B9FF", "Neurotrophic / regenerative": "#55EFC4",
    "Symptomatic / Other": "#B2BEC3", _INSUFFICIENT: "#DFE6E9",
}
_DEFAULT_COLOR = "#B2BEC3"
_STATUS_BADGE = {
    "recruiting": ("#00B894", "Recruiting"), "active": ("#0984E3", "Active"),
    "completed": ("#636E72", "Completed"), "terminated": ("#D63031", "Terminated"),
    "other": ("#B2BEC3", "Unknown"),
}


PHASE_OPTIONS = ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Not applicable"]


def _phase_buckets(phase: str) -> set:
    """Map a ClinicalTrials.gov phase string (incl. combos like 'PHASE1, PHASE2') to UI buckets."""
    p = (phase or "").upper()
    b = set()
    if "PHASE1" in p:  # also catches EARLY_PHASE1
        b.add("Phase 1")
    if "PHASE2" in p:
        b.add("Phase 2")
    if "PHASE3" in p:
        b.add("Phase 3")
    if "PHASE4" in p:
        b.add("Phase 4")
    return b or {"Not applicable"}  # NA / Expanded Access / blank


def _therapy_matches_phases(t: dict, selected: set | None) -> bool:
    """A therapy is shown if any of its trials falls in a selected phase. None/all = whole picture."""
    if not selected or len(selected) >= len(PHASE_OPTIONS):
        return True
    return any(_phase_buckets(tr.get("phase", "")) & selected for tr in t["trials"])


def load_landscape() -> dict | None:
    if not LANDSCAPE_PATH.exists():
        return None
    try:
        return json.loads(LANDSCAPE_PATH.read_text())
    except Exception:
        return None


def class_names(landscape: dict | None) -> list[str]:
    if not landscape:
        return []
    names = [c["name"] for c in landscape["classifications"]]
    if landscape.get("unclassified"):
        names.append(_INSUFFICIENT)
    return names


def _therapies_of(landscape: dict, class_name: str) -> list[dict]:
    if class_name == _INSUFFICIENT:
        return landscape.get("unclassified", [])
    for c in landscape["classifications"]:
        if c["name"] == class_name:
            return c["therapies"]
    return []


def therapy_labels(landscape: dict | None, class_name: str, phases: list | None = None) -> list[str]:
    """Dropdown labels, e.g. 'CNM-Au8 [contributing] — 5 trials, 3 recruiting'."""
    sel = set(phases) if phases else None
    labels = []
    for t in _therapies_of(landscape or {}, class_name):
        if not _therapy_matches_phases(t, sel):
            continue
        c = t["trial_counts"]
        rec = f", {c['recruiting']} recruiting" if c["recruiting"] else ""
        role = t.get("role_here")
        tag = f" [{role}]" if (role and class_name != _INSUFFICIENT) else ""
        labels.append(f"{t['name']}{tag} — {c['total']} trial{'s' if c['total'] != 1 else ''}{rec}")
    return labels


def _therapy_by_label(landscape: dict, class_name: str, label: str) -> dict | None:
    name = label.split(" — ")[0].split(" [")[0] if label else ""
    for t in _therapies_of(landscape, class_name):
        if t["name"] == name:
            return t
    return None


def build_sunburst(landscape: dict | None, phases: list | None = None):
    """Two-ring sunburst: inner = mechanism class, outer = its PRIMARY therapies (sized by trials).
    Contributing memberships live in the detail panel; abstentions get an 'Insufficient evidence' wedge.
    """
    import plotly.graph_objects as go

    sel = set(phases) if phases else None

    def _keep(t):
        return _therapy_matches_phases(t, sel)

    ids, labels, parents, values, colors, hover = [], [], [], [], [], []

    def add_class(name, therapies, color, desc=""):
        if not therapies:
            return
        cid = f"cls::{name}"
        total = sum(t["trial_counts"]["total"] for t in therapies)
        ids.append(cid); labels.append(name); parents.append(""); values.append(total); colors.append(color)
        hover.append(f"<b>{name}</b><br>{len(therapies)} therapies · {total} trials<br>{desc}")
        for t in therapies:
            cnt = t["trial_counts"]
            ids.append(f"th::{name}::{t['name']}"); labels.append(t["name"]); parents.append(cid)
            values.append(cnt["total"]); colors.append(color)
            hover.append(f"<b>{t['name']}</b> ({t['modality']})<br>Target: {t['target']}<br>"
                         f"{cnt['total']} trials · {cnt['recruiting']} recruiting · {cnt['completed']} completed")

    if landscape:
        for c in landscape["classifications"]:
            primaries = [t for t in c["therapies"] if t.get("role_here") == "primary" and _keep(t)]
            add_class(c["name"], primaries, _CLASS_COLORS.get(c["name"], _DEFAULT_COLOR), c.get("description", ""))
        un = [t for t in landscape.get("unclassified", []) if _keep(t)]
        if un:
            add_class(_INSUFFICIENT, un, _CLASS_COLORS[_INSUFFICIENT],
                      "Mechanism not established from available evidence")

    fig = go.Figure(go.Sunburst(
        ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
        marker=dict(colors=colors), hovertext=hover, hoverinfo="text",
        insidetextorientation="radial", maxdepth=2,
    ))
    fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=520, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def therapy_detail_md(landscape: dict | None, class_name: str, therapy_label: str) -> str:
    if not landscape or not class_name:
        return "*Select a mechanism class and therapy to see its mechanisms and trials.*"
    t = _therapy_by_label(landscape, class_name, therapy_label)
    if not t:
        return f"*{class_name}* — select a therapy above."
    header = f"### {t['name']}\n<sub>{t['modality']} · Target: {t['target']}</sub>"
    if t.get("aliases"):
        header += f"\n<sub>Also: {', '.join(t['aliases'][:6])}</sub>"
    mechs = t.get("mechanisms") or []
    if not mechs:
        return header + "\n\n**Mechanism of action:** _Not established from the available evidence._"
    lines = [header, "\n**Mechanism(s) of action:**"]
    for m in mechs:
        conf = int(round(m["confidence"] * 100))
        ev = f" — {m['evidence']}" if m.get("evidence") else ""
        lines.append(f"- **{m['class']}** · _{m['role']}, {conf}% confidence_{ev}")
    return "\n".join(lines)


def trials_table_html(landscape: dict | None, class_name: str, therapy_label: str) -> str:
    if not landscape or not class_name:
        return ""
    t = _therapy_by_label(landscape, class_name, therapy_label)
    if not t:
        return ""
    rows = []
    for tr in t["trials"]:
        color, label = _STATUS_BADGE.get(tr["status_group"], _STATUS_BADGE["other"])
        badge = (f'<span style="background:{color};color:#fff;border-radius:10px;'
                 f'padding:1px 8px;font-size:0.72rem;white-space:nowrap;">{label}</span>')
        phase = html.escape((tr.get("phase") or "—").replace("PHASE", "Ph"))
        title = html.escape(tr.get("title", "")[:110])
        nct = html.escape(tr.get("nct_id", ""))
        url = html.escape(tr.get("url", ""))
        sponsor = html.escape((tr.get("sponsor") or "")[:40])
        rows.append(
            f'<tr><td style="padding:4px 8px;">{badge}</td>'
            f'<td style="padding:4px 8px;color:#666;">{phase}</td>'
            f'<td style="padding:4px 8px;"><a href="{url}" target="_blank" rel="noopener">{nct}</a> — {title}</td>'
            f'<td style="padding:4px 8px;color:#888;font-size:0.8rem;">{sponsor}</td></tr>'
        )
    counts = t["trial_counts"]
    caption = (f'<div style="font-size:0.85rem;color:#666;margin:6px 0;">{counts["total"]} trials — '
               f'<b style="color:#00B894;">{counts["recruiting"]} recruiting</b>, '
               f'{counts["active"]} active, {counts["completed"]} completed, {counts["terminated"]} terminated</div>')
    return caption + (
        '<table style="width:100%;border-collapse:collapse;font-size:0.88rem;">'
        '<thead><tr style="text-align:left;border-bottom:1px solid #ddd;color:#888;">'
        '<th style="padding:4px 8px;">Status</th><th style="padding:4px 8px;">Phase</th>'
        '<th style="padding:4px 8px;">Trial</th><th style="padding:4px 8px;">Sponsor</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )
