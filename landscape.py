"""Rendering helpers for the Experimental ALS Therapy Landscape tab.

Loads data/landscape/landscape.json and renders a single "ALS Therapeutic Pipeline by
Clinical Trial Phase" wheel: mechanism groups are angular SECTORS, the three trial phases
are concentric RINGS (inner = Phase 1, outer = Phase 3), and each compound is a dot inside
its sector×ring cell (hover shows its name). Dots go grey when the compound has no
recruiting/active trial in the current view. Two dropdowns filter the wheel (recruitment
status + trial phase); a Mechanism dropdown surfaces a group's compounds, each with its
pipeline stage, mechanism confidence, and a trials table.
"""
from __future__ import annotations

import json
import html

from config import LANDSCAPE_PATH

_INSUFFICIENT = "Insufficient evidence"

_STATUS_BADGE = {
    "recruiting": ("#00B894", "Recruiting"), "active": ("#0984E3", "Active"),
    "completed": ("#636E72", "Completed"), "terminated": ("#D63031", "Terminated"),
    "other": ("#B2BEC3", "Unknown"),
}

# Recruitment-status filter for the pipeline wheel.
STATUS_FILTER_OPTIONS = ["All trials", "Recruiting", "Not recruiting"]
# Sentinel for the mechanism filter meaning "don't narrow the wheel to one group".
ALL_MECHANISMS = "All mechanisms"

# Pipeline-wheel rings, inner → outer. Phase 3 and Expanded Access (EAP) share the outer
# ring. These double as the "Trial phase" checkbox options. (Phase 4 / NA are never placed.)
PHASE_RINGS = ["Phase 1", "Phase 2", "Phase 3 / EAP"]

# Grey used for compounds with no recruiting/active trial in the current view.
_INACTIVE_COLOR = "#B8BFC7"


def _phase_buckets(phase: str) -> set:
    """Map a ClinicalTrials.gov phase string to wheel rings (Phase 3 + Expanded Access merge)."""
    p = (phase or "").upper()
    b = set()
    if "PHASE1" in p:  # also catches EARLY_PHASE1
        b.add("Phase 1")
    if "PHASE2" in p:
        b.add("Phase 2")
    if "PHASE3" in p or "EXPANDED" in p or "ACCESS" in p:
        b.add("Phase 3 / EAP")
    return b or {"Not applicable"}  # PHASE4 / NA / blank are not placed on the wheel


def _top_ring(trials: list[dict]) -> str | None:
    """Most advanced wheel ring (PHASE_RINGS order) present across a compound's trials."""
    present: set = set()
    for tr in trials:
        present |= _phase_buckets(tr.get("phase", ""))
    for ring in reversed(PHASE_RINGS):
        if ring in present:
            return ring
    return None


def load_landscape() -> dict | None:
    if not LANDSCAPE_PATH.exists():
        return None
    try:
        return json.loads(LANDSCAPE_PATH.read_text())
    except Exception:
        return None


# ── "ALS Therapeutic Pipeline by Clinical Trial Phase" wheel (inline SVG) ──────
# Magazine-style wheel: mechanism groups are equal angular SECTORS, the three
# trial phases are concentric RINGS (inner=Phase 1, outer=Phase 3), and each
# compound is a dot inside its sector×ring cell (hover shows its name). Driven by
# real landscape data; the app's 12 mechanism classes are mapped to 8 display groups.

def _all_compounds(landscape: dict) -> list[dict]:
    """Distinct compounds (therapies) across all classes + unclassified, deduped by name.

    A therapy is multi-label (appears under each class it acts through), but every copy
    carries the same `mechanisms` and `trials`, so keeping the first is sufficient.
    """
    seen: dict[str, dict] = {}
    for c in landscape.get("classifications", []):
        for t in c.get("therapies", []):
            seen.setdefault(t["name"], t)
    for t in landscape.get("unclassified", []):
        seen.setdefault(t["name"], t)
    return list(seen.values())


def _primary_class(therapy: dict) -> str:
    """The compound's dominant mechanism class (primary role, else highest confidence)."""
    mechs = therapy.get("mechanisms") or []
    if not mechs:
        return _INSUFFICIENT
    pool = [m for m in mechs if m.get("role") == "primary"] or mechs
    best = max(pool, key=lambda m: m.get("confidence", 0) or 0)
    return best.get("class", _INSUFFICIENT)


def _filter_trials(trials: list[dict], status_filter: str, phases: list | None = None) -> list[dict]:
    """Filter a compound's trials by recruitment status and the selected wheel rings.

    `phases` is a list of PHASE_RINGS labels (None = all rings). Trials that map only to a
    non-ring bucket (Phase 4 / NA) are always dropped — the wheel is Phase 1–3/EAP only.
    """
    out = trials
    if status_filter == "Recruiting":
        out = [tr for tr in out if tr.get("status_group") == "recruiting"]
    elif status_filter == "Not recruiting":
        out = [tr for tr in out if tr.get("status_group") != "recruiting"]
    selected = set(phases) if phases else set(PHASE_RINGS)
    out = [tr for tr in out if _phase_buckets(tr.get("phase", "")) & selected]
    return list(out)


def _compound_active(trials: list[dict]) -> bool:
    """True if any trial is recruiting or active-not-recruiting (drives dot coloring)."""
    return any(tr.get("status_group") in ("recruiting", "active") for tr in trials)


# Eight display groups (clockwise from top) and their colors, matching the
# reference infographic. The app's finer 12-class taxonomy maps down to these.
GROUP_ORDER = [
    "Neuroinflammation / Immunity",
    "RNA / Gene Targeting",
    "Neuroprotection / Cell Survival",
    "Protein Homeostasis / TDP-43 Pathology",
    "Metabolic / Mitochondrial Function",
    "Neuromuscular Function",
    "Stem Cell / Regenerative",
    "Others / Multiple",
]
GROUP_COLORS = {
    "Neuroinflammation / Immunity": "#4C6FB1",
    "RNA / Gene Targeting": "#E4586E",
    "Neuroprotection / Cell Survival": "#26B6A6",
    "Protein Homeostasis / TDP-43 Pathology": "#EFAA3A",
    "Metabolic / Mitochondrial Function": "#9B7EC8",
    "Neuromuscular Function": "#EE7B4E",
    "Stem Cell / Regenerative": "#5BB56A",
    "Others / Multiple": "#F4CE14",  # yellow (grey is reserved for inactive compounds)
}
_CLASS_TO_GROUP = {
    "TDP-43 proteinopathy": "Protein Homeostasis / TDP-43 Pathology",
    "Proteostasis / autophagy": "Protein Homeostasis / TDP-43 Pathology",
    "SOD1": "RNA / Gene Targeting",
    "C9orf72": "RNA / Gene Targeting",
    "FUS": "RNA / Gene Targeting",
    "RNA metabolism": "RNA / Gene Targeting",
    "Neuroinflammation": "Neuroinflammation / Immunity",
    "Oxidative stress": "Neuroprotection / Cell Survival",
    "Glutamate excitotoxicity": "Neuroprotection / Cell Survival",
    "Mitochondrial dysfunction": "Metabolic / Mitochondrial Function",
    "Neurotrophic / regenerative": "Stem Cell / Regenerative",
    "Symptomatic / Other": "Others / Multiple",
    _INSUFFICIENT: "Others / Multiple",
}

_DOT_CAP = 12  # max dots drawn per sector×ring cell (real counts live in hover/summary)


def _group_of(therapy: dict) -> str:
    return _CLASS_TO_GROUP.get(_primary_class(therapy), "Others / Multiple")


def _pipeline_grid(landscape: dict | None, status_filter: str, phases: list,
                   mech_filter: str = ALL_MECHANISMS) -> dict:
    """{group: {ring: [(compound name, is_active)]}} over the filtered, ring-placed compounds.

    `phases` is the list of selected PHASE_RINGS; `mech_filter` other than ALL_MECHANISMS
    narrows the wheel to a single mechanism group. Each compound lands in the most advanced
    selected ring it has a trial in.
    """
    grid = {g: {r: [] for r in PHASE_RINGS} for g in GROUP_ORDER}
    if landscape:
        for t in _all_compounds(landscape):
            group = _group_of(t)
            if mech_filter != ALL_MECHANISMS and group != mech_filter:
                continue
            trials = _filter_trials(t.get("trials", []), status_filter, phases)
            if not trials:
                continue
            ring = _top_ring(trials)
            if not ring:
                continue
            grid[group][ring].append((t["name"], _compound_active(trials)))
    return grid


def _polar_xy(cx: float, cy: float, r: float, ang_deg: float) -> tuple:
    """Angle 0 = top (12 o'clock), increasing clockwise, in screen coords."""
    import math
    t = math.radians(ang_deg - 90.0)
    return cx + r * math.cos(t), cy + r * math.sin(t)


def _annular_sector_path(cx, cy, r_in, r_out, a0, a1) -> str:
    large = 1 if (a1 - a0) % 360 > 180 else 0
    x0o, y0o = _polar_xy(cx, cy, r_out, a0)
    x1o, y1o = _polar_xy(cx, cy, r_out, a1)
    x1i, y1i = _polar_xy(cx, cy, r_in, a1)
    x0i, y0i = _polar_xy(cx, cy, r_in, a0)
    return (f"M {x0o:.1f} {y0o:.1f} A {r_out:.1f} {r_out:.1f} 0 {large} 1 {x1o:.1f} {y1o:.1f} "
            f"L {x1i:.1f} {y1i:.1f} A {r_in:.1f} {r_in:.1f} 0 {large} 0 {x0i:.1f} {y0i:.1f} Z")


def _cell_dots(cx, cy, r_in, r_out, a0, a1, cells, col) -> str:
    """Lay up to _DOT_CAP compound dots on a jittered grid inside one annular cell.

    `cells` is a list of (name, is_active); active dots take the group `col`, inactive grey.
    """
    import math
    n = min(len(cells), _DOT_CAP)
    if n == 0:
        return ""
    cols = 4 if n > 6 else max(1, min(n, 3))
    rows = max(1, math.ceil(n / cols))
    rr0, rr1 = r_in + 13, r_out - 13
    aa0, aa1 = a0 + 3.0, a1 - 3.0
    out = []
    for k, (name, active) in enumerate(cells[:n]):
        row, col_i = divmod(k, cols)
        in_row = min(cols, n - row * cols)
        fr = (row + 0.5) / rows
        fa = (col_i + 0.5) / in_row
        jr = ((k * 37) % 7 - 3) * 1.1
        ja = ((k * 53) % 5 - 2) * 0.6
        r = rr0 + fr * (rr1 - rr0) + jr
        a = aa0 + fa * (aa1 - aa0) + ja
        x, y = _polar_xy(cx, cy, r, a)
        fill = col if active else _INACTIVE_COLOR
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{fill}" '
                   f'stroke="#fff" stroke-width="1.3"><title>{html.escape(name)}</title></circle>')
    return "".join(out)


def build_pipeline_svg(landscape: dict | None, status_filter: str = "All trials",
                       phases: list | None = None,
                       mech_filter: str = ALL_MECHANISMS) -> str:
    """Inline-SVG 'ALS Therapeutic Pipeline by Phase' infographic (see section header).

    `phases` (list of PHASE_RINGS labels; None = all) drives BOTH the filter and the geometry:
    one concentric ring is drawn per selected phase, inner → outer in PHASE_RINGS order.
    """
    rings = [r for r in PHASE_RINGS if r in set(phases)] if phases else list(PHASE_RINGS)
    grid = _pipeline_grid(landscape, status_filter, rings, mech_filter)
    groups = [g for g in GROUP_ORDER if any(grid[g][r] for r in rings)]
    ph_tot = {r: sum(len(grid[g][r]) for g in groups) for r in rings}
    total = sum(ph_tot.values())

    if not groups or total == 0 or not rings:
        sel = ", ".join(rings) if rings else "no phases"
        return ('<div style="padding:40px;text-align:center;color:#888;font-size:15px;">'
                f'No trials match “{html.escape(status_filter)} · {html.escape(sel)} · '
                f'{html.escape(mech_filter)}”.</div>')

    W = 720
    cx = cy = W / 2
    r_hub = 78
    r_out = 338
    nr = len(rings)
    t = (r_out - r_hub) / nr  # ring thickness recomputed for the number of selected phases
    n = len(groups)
    seg = 360.0 / n
    gap_a = 1.4

    mx, my_t, my_b = 116, 46, 70  # margins so perimeter labels aren't clipped
    svg = [f'<svg viewBox="{-mx} {-my_t} {W + 2 * mx} {W + my_t + my_b}" width="100%" '
           f'style="max-width:840px;height:auto;" '
           'font-family="-apple-system,Segoe UI,Roboto,sans-serif">']

    labels = []  # perimeter category + exemplar labels, drawn after wedges
    for gi, g in enumerate(groups):
        col = GROUP_COLORS[g]
        center = gi * seg  # group 0 centered at top
        a0, a1 = center - seg / 2 + gap_a, center + seg / 2 - gap_a
        for i, ring in enumerate(rings):
            ri = r_hub + i * t + 2.5
            ro = r_hub + (i + 1) * t - 2.5
            cells = sorted(grid[g][ring], key=lambda c: c[0])
            cnt = len(cells)
            path = _annular_sector_path(cx, cy, ri, ro, a0, a1)
            fillop = 0.13 + 0.05 * i
            svg.append(f'<path d="{path}" fill="{col}" fill-opacity="{fillop:.2f}" '
                       f'stroke="#fff" stroke-width="2"><title>{html.escape(g)} — {html.escape(ring)}: '
                       f'{cnt} compound{"s" if cnt != 1 else ""}</title></path>')
            svg.append(_cell_dots(cx, cy, ri, ro, a0, a1, cells, col))

        # perimeter label: category name (wrapped at " / ") + count + top exemplar
        exemplar = next((sorted(grid[g][r], key=lambda c: c[0])[0][0]
                         for r in reversed(rings) if grid[g][r]), "")
        lx, ly = _polar_xy(cx, cy, r_out + 22, center)
        if 15 < center < 165:
            anchor, x = "start", lx + 4
        elif 195 < center < 345:
            anchor, x = "end", lx - 4
        else:
            anchor, x = "middle", lx
        gcnt = sum(len(grid[g][r]) for r in rings)
        parts = [html.escape(s) for s in g.split(" / ")]
        cnt_tspan = f'<tspan font-weight="400" fill="#98a0aa"> ({gcnt})</tspan>'
        common = f'x="{x:.1f}" text-anchor="{anchor}" font-size="12.5" font-weight="700" fill="{col}"'
        if len(parts) >= 2:
            lab = (f'<text {common} y="{ly:.1f}">{parts[0]} /</text>'
                   f'<text {common} y="{ly + 14:.1f}">{" / ".join(parts[1:])}{cnt_tspan}</text>')
            yy = ly + 29
        else:
            lab = f'<text {common} y="{ly:.1f}">{parts[0]}{cnt_tspan}</text>'
            yy = ly + 15
        if exemplar:
            lab += (f'<text x="{x:.1f}" y="{yy:.1f}" text-anchor="{anchor}" '
                    f'font-size="10.5" fill="#8a929c">e.g. {html.escape(exemplar[:20])}</text>')
        labels.append(lab)

    svg.extend(labels)

    # center hub
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r_hub - 6}" fill="#fff" stroke="#dde5ef" stroke-width="2"/>')
    svg.append(f'<text x="{cx}" y="{cy - 4}" text-anchor="middle" font-size="27" font-weight="800" '
               f'fill="#2b3a4a" letter-spacing="1">ALS</text>')
    svg.append(f'<text x="{cx}" y="{cy + 15}" text-anchor="middle" font-size="10.5" '
               f'fill="#6b7683">Therapeutic Pipeline</text>')
    svg.append(f'<text x="{cx}" y="{cy + 34}" text-anchor="middle" font-size="15">🧬</text>')

    # ring labels, stacked at top with a white halo for legibility
    for i, ring in enumerate(rings):
        rmid = r_hub + (i + 0.5) * t
        _, y = _polar_xy(cx, cy, rmid, 0)
        svg.append(f'<text x="{cx}" y="{y - 3:.1f}" text-anchor="middle" font-size="13.5" '
                   f'font-weight="800" fill="#3a4a5a" stroke="#fff" stroke-width="3.2" '
                   f'paint-order="stroke" style="paint-order:stroke">{html.escape(ring)}</text>')
        svg.append(f'<text x="{cx}" y="{y + 12:.1f}" text-anchor="middle" font-size="11.5" '
                   f'font-weight="700" fill="#5a6675" stroke="#fff" stroke-width="3" '
                   f'paint-order="stroke" style="paint-order:stroke">({ph_tot[ring]})</text>')
    svg.append("</svg>")

    # ── side panels (legend + summary) ──
    legend_rows = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;font-size:12px;color:#3a4453;">'
        f'<span style="width:11px;height:11px;border-radius:50%;background:{GROUP_COLORS[g]};'
        f'flex:0 0 auto;"></span><span>{html.escape(g)}</span>'
        f'<span style="margin-left:auto;color:#98a0aa;">{sum(len(grid[g][r]) for r in rings)}</span></div>'
        for g in groups)
    legend_rows += (
        '<div style="display:flex;align-items:center;gap:8px;margin:5px 0;font-size:12px;color:#3a4453;'
        'border-top:1px solid #eef2f6;padding-top:6px;">'
        f'<span style="width:11px;height:11px;border-radius:50%;background:{_INACTIVE_COLOR};'
        'flex:0 0 auto;"></span><span>Inactive — no recruiting/active trial</span></div>')
    legend = (
        '<div style="border:1px solid #e6ebf1;border-radius:12px;padding:12px 14px;background:#fff;">'
        '<div style="font-weight:700;color:#2b3a4a;font-size:13px;margin-bottom:6px;">Mechanism of Action</div>'
        f'{legend_rows}</div>')

    def _row(lbl, val, sub, strong=False):
        w = "800" if strong else "600"
        return (f'<div style="display:flex;justify-content:space-between;align-items:baseline;'
                f'padding:7px 0;border-top:1px solid #eef2f6;">'
                f'<span style="color:#3a4453;font-weight:{w};font-size:13px;">{lbl}</span>'
                f'<span style="text-align:right;"><b style="font-size:15px;color:#2b3a4a;">{val}</b>'
                f'<span style="display:block;font-size:10.5px;color:#98a0aa;">{sub}</span></span></div>')

    pct = lambda v: f"{round(100 * v / total)}%" if total else "0%"
    summary = (
        '<div style="border:1px solid #e6ebf1;border-radius:12px;padding:12px 14px;background:#fff;margin-top:12px;">'
        '<div style="font-weight:700;color:#2b3a4a;font-size:13px;text-align:center;margin-bottom:2px;">Pipeline Summary</div>'
        + "".join(_row(r, ph_tot[r], pct(ph_tot[r])) for r in rings)
        + _row("Total", total, "Candidates", strong=True)
        + f'<div style="margin-top:8px;font-size:10.5px;color:#98a0aa;text-align:center;">'
          f'Showing: {html.escape(status_filter)} · {html.escape(mech_filter)}</div></div>')

    subtitle = " &nbsp;|&nbsp; ".join(
        f'{"Inner" if i == 0 else "Outer" if i == nr - 1 else "Middle"} ring: {html.escape(r)}'
        for i, r in enumerate(rings))
    return (
        '<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;">'
        '<div style="text-align:center;margin-bottom:4px;">'
        '<div style="font-size:21px;font-weight:800;color:#22303f;">ALS Therapeutic Pipeline by Clinical Trial Phase</div>'
        f'<div style="font-size:13px;color:#8a929c;margin-top:2px;">{subtitle}</div></div>'
        '<div style="display:flex;gap:18px;align-items:flex-start;justify-content:center;flex-wrap:wrap;">'
        f'<div style="flex:1 1 460px;min-width:340px;max-width:720px;">{"".join(svg)}</div>'
        f'<div style="flex:0 0 232px;width:232px;">{legend}{summary}</div>'
        '</div></div>')


# ── Mechanism → compound drill-down (drives the detail panel + trials table) ───

def group_names(landscape: dict | None) -> list[str]:
    """Display groups that have at least one compound, in canonical GROUP_ORDER."""
    if not landscape:
        return []
    present = {_group_of(t) for t in _all_compounds(landscape)}
    return [g for g in GROUP_ORDER if g in present]


def mechanism_filter_options(landscape: dict | None) -> list[str]:
    """Mechanism dropdown choices: 'All mechanisms' plus every group that has a compound."""
    return [ALL_MECHANISMS] + group_names(landscape)


def compounds_of_group(landscape: dict | None, group: str,
                       status_filter: str = "All trials", phases: list | None = None) -> list[dict]:
    """Compounds in `group` (or all groups when group is ALL_MECHANISMS) with ≥1 passing trial."""
    out = []
    for t in _all_compounds(landscape or {}):
        if group and group != ALL_MECHANISMS and _group_of(t) != group:
            continue
        if not _filter_trials(t.get("trials", []), status_filter, phases):
            continue
        out.append(t)
    return out


def compound_labels(landscape: dict | None, group: str,
                    status_filter: str = "All trials", phases: list | None = None) -> list[str]:
    """Dropdown labels, e.g. 'Tofersen — 3 trials, 1 recruiting'."""
    labels = []
    for t in compounds_of_group(landscape, group, status_filter, phases):
        c = t["trial_counts"]
        rec = f", {c['recruiting']} recruiting" if c["recruiting"] else ""
        labels.append(f"{t['name']} — {c['total']} trial{'s' if c['total'] != 1 else ''}{rec}")
    return labels


def _compound_by_label(landscape: dict | None, label: str) -> dict | None:
    name = label.split(" — ")[0] if label else ""
    if not name:
        return None
    for t in _all_compounds(landscape or {}):
        if t["name"] == name:
            return t
    return None


def compound_detail_md(landscape: dict | None, label: str) -> str:
    if not landscape:
        return "*Select a compound to see its pipeline stage and mechanisms.*"
    t = _compound_by_label(landscape, label)
    if not t:
        return "*Select a compound above.*"
    stage = _top_ring(t["trials"]) or "Preclinical / not applicable"
    header = (f"### {t['name']}\n<sub>{t['modality']} · Target: {t['target']} · "
              f"Pipeline stage: {stage}</sub>")
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


def compound_trials_html(landscape: dict | None, label: str) -> str:
    if not landscape:
        return ""
    t = _compound_by_label(landscape, label)
    if not t:
        return ""
    rows = []
    for tr in t["trials"]:
        color, badge_label = _STATUS_BADGE.get(tr["status_group"], _STATUS_BADGE["other"])
        badge = (f'<span style="background:{color};color:#fff;border-radius:10px;'
                 f'padding:1px 8px;font-size:0.72rem;white-space:nowrap;">{badge_label}</span>')
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
