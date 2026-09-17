"""Facility / geography trial search + research-evidence enrichment.

Physicians ask "what trials are at Mass General Hospital" or "what trials are in NY".
This module filters the offline-ingested trials list by facility/city/state/country
(in-memory — no fetch at query time) and enriches each hit with surrounding research
evidence: key papers, sibling trials for the same compound, a heuristic evidence-strength
tier, and recruiting status.
"""
from __future__ import annotations

import html
import json
from functools import lru_cache
from typing import TYPE_CHECKING

from logging_config import get_logger

if TYPE_CHECKING:
    import chromadb
    import networkx as nx

_logger = get_logger("trials_query")

# Site statuses that count as "recruiting / open to enrollment".
RECRUITING_STATUSES = {
    "RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION", "AVAILABLE",
}

# Trial status display order — open first, closed/unavailable last (mirrors research_agent).
_TRIAL_STATUS_RANK = {
    "AVAILABLE": 0, "RECRUITING": 1, "NOT_YET_RECRUITING": 2, "ENROLLING_BY_INVITATION": 3,
    "ACTIVE_NOT_RECRUITING": 4, "TEMPORARILY_NOT_AVAILABLE": 5, "COMPLETED": 6,
    "SUSPENDED": 7, "TERMINATED": 8, "WITHDRAWN": 9, "NO_LONGER_AVAILABLE": 10,
}

# US state abbreviation → full name. ClinicalTrials.gov stores the full state name, so
# "NY" must be normalized to "New York" before matching.
_STATE_ABBREV = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
    "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
    "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
    "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia", "WA": "Washington",
    "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

# Generic facility words that carry no discriminating signal on their own.
_FACILITY_STOPWORDS = {"of", "the", "and", "at", "for"}


def _norm_alnum(s: str) -> str:
    """Lowercase, alphanumeric-only form so 'CNM-Au8', 'CNMAu8', 'cnm_au8' all unify."""
    return "".join(c for c in s.lower() if c.isalnum())


def _tokens(s: str) -> list[str]:
    """Lowercased alphanumeric word tokens, minus stopwords."""
    words = "".join(c if c.isalnum() else " " for c in s.lower()).split()
    return [w for w in words if w not in _FACILITY_STOPWORDS]


def _normalize_state(state: str) -> str:
    """Map a state abbreviation to its full name; pass full names through unchanged."""
    s = state.strip()
    return _STATE_ABBREV.get(s.upper(), s)


def _facility_matches(query: str, site_facility: str) -> bool:
    """True if every query token is a substring of some site-facility token.

    Token-level substring matching lets 'Mass General' match 'Massachusetts General
    Hospital' ('mass' ⊂ 'massachusetts') without a fuzzy library.
    """
    q_tokens = _tokens(query)
    if not q_tokens:
        return False
    site_tokens = _tokens(site_facility)
    return all(any(q in st for st in site_tokens) for q in q_tokens)


def _site_matches(
    site: dict,
    facility: str | None,
    city: str | None,
    state: str | None,
    country: str | None,
) -> bool:
    """True if a single trial site satisfies every provided location filter."""
    if facility and not _facility_matches(facility, site.get("facility", "")):
        return False
    if city and city.strip().lower() not in (site.get("city", "") or "").lower():
        return False
    if state:
        want = _normalize_state(state).lower()
        if want != (site.get("state", "") or "").lower():
            return False
    if country and country.strip().lower() not in (site.get("country", "") or "").lower():
        return False
    return True


def search_trials_by_location(
    trials: list[dict],
    *,
    facility: str | None = None,
    city: str | None = None,
    state: str | None = None,
    country: str | None = None,
    status: str | None = None,
    study_type: str | None = None,
) -> list[dict]:
    """Return trials with at least one site matching the location filters.

    `status`: "Recruiting" keeps only trials with an open overall status; "Not recruiting"
    keeps only closed ones; anything else (None / "All") keeps all.
    `study_type`: "Interventional" keeps interventional trials; "Expanded Access" keeps
    expanded-access (investigational-use) programs; anything else (None / "All") keeps both.
    Each returned trial is a shallow copy with `matched_sites` attached; recruiting first.
    """
    if not any([facility, city, state, country]):
        return []

    status_filter = (status or "").strip().lower()
    type_filter = (study_type or "").strip().lower()
    results: list[dict] = []
    for trial in trials:
        is_eap = bool(trial.get("is_expanded_access"))
        if type_filter == "interventional" and is_eap:
            continue
        if type_filter == "expanded access" and not is_eap:
            continue

        matched_sites = [
            s for s in trial.get("locations", [])
            if _site_matches(s, facility, city, state, country)
        ]
        if not matched_sites:
            continue

        is_recruiting = trial.get("status", "") in RECRUITING_STATUSES
        if status_filter == "recruiting" and not is_recruiting:
            continue
        if status_filter == "not recruiting" and is_recruiting:
            continue

        hit = dict(trial)
        hit["matched_sites"] = matched_sites
        results.append(hit)

    results.sort(key=lambda t: _TRIAL_STATUS_RANK.get(t.get("status", ""), 99))
    return results


# ── Autocomplete vocabulary for the facility / city combobox inputs ───────────
# The facility/city fields are typeable comboboxes (gr.Dropdown, filterable): the physician
# types and PICKS from the attached list, rather than the system guessing from a substring
# (where "new" is ambiguously New York / New Haven / Newport Beach…). Choices are the names
# actually present in the trials, ranked by trial-site count (busiest first) so the highest-
# yield option surfaces first — for cities this is the practical stand-in for "population" and
# also covers international cities. Gradio filters the preloaded list client-side as you type.

MIN_AUTOCOMPLETE_CHARS = 3  # client-side gate: the attached list stays hidden until this many chars


def build_location_index(trials: list[dict]) -> dict[str, list[tuple[str, int]]]:
    """Distinct facility + city names with their TRIAL counts, each sorted busiest first.

    Counts distinct trials, not site rows: a single trial that lists an anonymized placeholder
    like "GSK Investigational Site" 49 times counts once, so placeholders don't balloon to the
    top of the ranking and the shown count matches what a search can return. Built once at
    startup; feeds location_choices() which becomes the combobox `choices`.
    """
    from collections import Counter

    city_counts: Counter = Counter()
    facility_counts: Counter = Counter()
    for t in trials:
        cities_here: set[str] = set()
        facilities_here: set[str] = set()
        for s in t.get("locations", []):
            city = (s.get("city") or "").strip()
            facility = (s.get("facility") or "").strip()
            if city:
                cities_here.add(city)
            if facility:
                facilities_here.add(facility)
        for c in cities_here:
            city_counts[c] += 1
        for f in facilities_here:
            facility_counts[f] += 1

    def _ranked(counter: Counter) -> list[tuple[str, int]]:
        # busiest first, then alphabetical for stable ties
        return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0].lower()))

    return {"cities": _ranked(city_counts), "facilities": _ranked(facility_counts)}


def location_choices(index: dict) -> dict[str, list[str]]:
    """Gradio combobox choices — plain names, busiest first.

    Just the names (no "· N trials" suffix): with a filterable/custom-value Dropdown the
    displayed option text becomes the field value, so any suffix would leak into the search
    term. Ranking is preserved by list order; the count is used only for that ordering.
    """
    return {
        "cities": [name for name, _ in index.get("cities", [])],
        "facilities": [name for name, _ in index.get("facilities", [])],
    }


def _find_supporting_papers(
    trial: dict,
    collection: "chromadb.Collection | None",
    top_n: int = 3,
) -> list[dict]:
    """Retrieve the top research papers relevant to this trial's compound/target."""
    if collection is None or collection.count() == 0:
        return []

    from rag import retriever as rag_retriever

    entities = list(trial.get("target_entities") or [])
    if entities:
        results = rag_retriever.search_by_entities(collection, entities)
    else:
        # No extracted target — fall back to the trial title + intervention names.
        iv_names = " ".join(iv.get("name", "") for iv in trial.get("interventions", []))
        query = f"{trial.get('title', '')} {iv_names}".strip()
        results = rag_retriever.search(collection, query) if query else []

    results = rag_retriever.apply_citation_boost(results)
    return [
        {
            "pmid": r["pmid"],
            "title": r["title"],
            "year": r["year"],
            "citation_count": r["citation_count"],
        }
        for r in results[:top_n]
    ]


def _find_sibling_trials(
    trial: dict,
    all_trials: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """Other trials testing the same compound/target (prior + concurrent), excluding self."""
    self_nct = trial.get("nct_id", "")
    self_targets = {t.lower() for t in trial.get("target_entities", [])}
    self_ivs = {_norm_alnum(iv.get("name", "")) for iv in trial.get("interventions", [])}
    self_ivs.discard("")

    siblings: list[dict] = []
    for other in all_trials:
        if other.get("nct_id", "") == self_nct:
            continue
        other_targets = {t.lower() for t in other.get("target_entities", [])}
        other_ivs = {_norm_alnum(iv.get("name", "")) for iv in other.get("interventions", [])}
        if (self_targets & other_targets) or (self_ivs & other_ivs):
            siblings.append({
                "nct_id": other.get("nct_id", ""),
                "title": other.get("title", ""),
                "phase": other.get("phase", ""),
                "status": other.get("status", ""),
                "url": other.get("url", ""),
            })

    siblings.sort(key=lambda t: _TRIAL_STATUS_RANK.get(t.get("status", ""), 99))
    return siblings[:top_n]


# ── Targeted mechanism (reverse-indexed from the offline therapy landscape) ───
# A trial's "targeted mechanism" is the dominant mechanism class of the compound it
# tests, reused from the already-built landscape.json (no LLM at query time). Only
# landscape-classified compounds get one; unclassified drugs (e.g. brand-new agents
# whose only extracted target is their own name) map to "" and show no mechanism.

def _primary_class(therapy: dict) -> str:
    """A compound's dominant mechanism class: primary role, else highest confidence.

    Mirrors landscape._primary_class. Returns "" when the compound has no mechanism
    (landscape left it unclassified).
    """
    mechs = therapy.get("mechanisms") or []
    if not mechs:
        return ""
    pool = [m for m in mechs if m.get("role") == "primary"] or mechs
    best = max(pool, key=lambda m: m.get("confidence", 0) or 0)
    return best.get("class", "") or ""


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def _mechanism_index() -> dict[str, str]:
    """Map NCT ID → targeted-mechanism class name, reverse-indexed from landscape.json.

    Cached: landscape.json is static at runtime, so parse it once — not once per
    enriched trial (was 25× disk-read + JSON-parse per Clinical Trials search).

    Classified compounds take precedence over unclassified ones; compounds with no
    mechanism are skipped. Returns {} when the landscape artifact is absent.
    """
    from config import LANDSCAPE_PATH

    if not LANDSCAPE_PATH.exists():
        return {}
    try:
        landscape = json.loads(LANDSCAPE_PATH.read_text())
    except Exception:
        _logger.warning("could not load landscape for mechanism index", exc_info=True)
        return {}

    index: dict[str, str] = {}
    # classifications first so a classified compound's mechanism wins over unclassified
    for group in (landscape.get("classifications", []) + landscape.get("unclassified", [])):
        therapies = group.get("therapies", [group]) if "therapies" in group else [group]
        for therapy in therapies:
            mechanism = _primary_class(therapy)
            if not mechanism:
                continue
            for tr in therapy.get("trials", []):
                index.setdefault(tr.get("nct_id", ""), mechanism)
    index.pop("", None)
    return index


def _kg_paper_count(
    trial: dict, graph: "nx.DiGraph | None", mechanism: str = ""
) -> tuple[int, str]:
    """Max supporting-paper count in the KG, and the node label that supplied it.

    Considers this trial's target entities plus its targeted `mechanism` hub node, so a
    compound whose only extracted target is its own name (no KG node) still gets credit
    for its mechanism's literature instead of scoring 0. The returned label lets the tier
    show a physician *which* node the count came from (e.g. the mechanism vs. the target).
    """
    if graph is None:
        return 0, ""
    from graph.query import _find_node

    # (name, is_mechanism) — mechanism last so a real target wins ties.
    names = [(n, False) for n in trial.get("target_entities", [])]
    if mechanism:
        names.append((mechanism, True))

    best, best_label = 0, ""
    for name, is_mech in names:
        for node_id in _find_node(graph, name):
            count = graph.nodes[node_id].get("paper_count", 0)
            if count > best:
                best = count
                display = graph.nodes[node_id].get("display_name", name)
                best_label = f"{display} (mechanism)" if is_mech else display
    return best, best_label


def _evidence_tier(
    trial: dict,
    key_papers: list[dict],
    kg_paper_count: int,
    kg_source: str = "",
) -> dict:
    """Heuristic evidence-strength tier from paper count, citations, KG breadth, and phase.

    Strong / Moderate / Emerging — a fully offline signal (no LLM call). Returns a
    `rationale`: one entry per scoring factor with the points it earned and a physician-
    readable reason (including *which* KG node supplied the paper count), so the tier can
    explain itself rather than presenting a bare label.
    """
    n_papers = len(key_papers)
    max_cit = max((p.get("citation_count", 0) for p in key_papers), default=0)
    phase = (trial.get("phase", "") or "").upper()

    rationale: list[dict] = []

    def factor(label: str, pts: int, detail: str) -> None:
        rationale.append({"factor": label, "points": pts, "detail": detail})

    if n_papers >= 5:
        factor("Supporting papers", 2, f"{n_papers} supporting papers in the database (≥5)")
    elif n_papers >= 2:
        factor("Supporting papers", 1, f"{n_papers} supporting papers in the database (2–4)")
    else:
        factor("Supporting papers", 0, f"{n_papers} supporting paper(s) in the database (<2)")

    if max_cit >= 100:
        factor("Citation impact", 2, f"top paper cited {max_cit}× (≥100)")
    elif max_cit >= 20:
        factor("Citation impact", 1, f"top paper cited {max_cit}× (20–99)")
    else:
        factor("Citation impact", 0, f"top paper cited {max_cit}× (<20)")

    if kg_paper_count >= 5:
        src = f" via {kg_source}" if kg_source else ""
        factor("KG breadth", 1, f"{kg_paper_count} papers on the target/mechanism node{src} (≥5)")
    else:
        src = f" (best: {kg_source})" if kg_source else ""
        factor("KG breadth", 0, f"{kg_paper_count} papers on the target/mechanism node{src} (<5)")

    is_phase3 = "PHASE3" in phase.replace(" ", "") or "3" in phase
    if is_phase3:
        factor("Trial phase", 1, f"reached Phase 3 ({trial.get('phase', '')})")
    else:
        factor("Trial phase", 0, f"not yet Phase 3 ({trial.get('phase', '') or 'phase unknown'})")

    points = sum(f["points"] for f in rationale)

    if points >= 4:
        tier = "Strong"
    elif points >= 2:
        tier = "Moderate"
    else:
        tier = "Emerging"

    return {
        "tier": tier,
        "points": points,
        "n_papers": n_papers,
        "max_citations": max_cit,
        "kg_paper_count": kg_paper_count,
        "kg_source": kg_source,
        "rationale": rationale,
    }


def enrich_trial(
    trial: dict,
    collection: "chromadb.Collection | None" = None,
    graph: "nx.DiGraph | None" = None,
    all_trials: list[dict] | None = None,
    mechanism_index: dict[str, str] | None = None,
) -> dict:
    """Attach research-evidence context to a trial: targeted mechanism, key papers,
    sibling trials, evidence tier."""
    index = _mechanism_index() if mechanism_index is None else mechanism_index
    mechanism = index.get(trial.get("nct_id", ""), "")

    key_papers = _find_supporting_papers(trial, collection)
    sibling_trials = _find_sibling_trials(trial, all_trials or [])
    kg_paper_count, kg_source = _kg_paper_count(trial, graph, mechanism)
    evidence = _evidence_tier(trial, key_papers, kg_paper_count, kg_source)

    return {
        "nct_id": trial.get("nct_id", ""),
        "title": trial.get("title", ""),
        "phase": trial.get("phase", ""),
        "status": trial.get("status", ""),
        "is_recruiting": trial.get("status", "") in RECRUITING_STATUSES,
        "sponsor": trial.get("sponsor", ""),
        "url": trial.get("url", ""),
        "target_entities": trial.get("target_entities", []),
        "mechanism": mechanism,
        "mechanism_summary": trial.get("mechanism_summary", {}) or {},
        "eligibility": trial.get("eligibility", {}) or {},
        "matched_sites": trial.get("matched_sites", []),
        "key_papers": key_papers,
        "sibling_trials": sibling_trials,
        "evidence": evidence,
    }


# ── HTML rendering for the Clinical Trials tab ────────────────────────────────

def _status_badge(status: str) -> tuple[str, str]:
    """(color, label) for a ClinicalTrials.gov overall-status value."""
    s = (status or "").upper()
    if s in RECRUITING_STATUSES:
        return "#00B894", "Recruiting" if s == "RECRUITING" else status.replace("_", " ").title()
    if s == "ACTIVE_NOT_RECRUITING":
        return "#0984E3", "Active"
    if s == "COMPLETED":
        return "#636E72", "Completed"
    if s in {"TERMINATED", "WITHDRAWN", "SUSPENDED"}:
        return "#D63031", status.title()
    return "#B2BEC3", (status or "Unknown").replace("_", " ").title()


_TIER_COLOR = {"Strong": "#00B894", "Moderate": "#E17055", "Emerging": "#B2BEC3"}


def _pill(text: str, color: str) -> str:
    return (f'<span style="background:{color};color:#fff;border-radius:10px;'
            f'padding:1px 8px;font-size:0.72rem;white-space:nowrap;">{html.escape(text)}</span>')


def _tier_rationale_html(ev: dict) -> str:
    """A collapsible 'why this tier' breakdown: each scoring factor, its points, and reason.

    Uses a native <details> disclosure (no JS) so a physician can audit the label without
    it crowding the card by default.
    """
    rationale = ev.get("rationale") or []
    if not rationale:
        return ""
    rows = "".join(
        f'<li style="margin:1px 0;{"" if f["points"] else "color:#aaa;"}">'
        f'<b>+{f["points"]}</b> {html.escape(f["factor"])} — {html.escape(f["detail"])}</li>'
        for f in rationale
    )
    total = ev.get("points", sum(f["points"] for f in rationale))
    summary = (f'Why {html.escape(ev.get("tier", ""))}? ({total} pts — '
               "≥4 Strong · 2–3 Moderate · &lt;2 Emerging)")
    return (
        '<details style="margin-top:4px;font-size:0.8rem;color:#555;">'
        f'<summary style="cursor:pointer;color:#6C5CE7;">{summary}</summary>'
        f'<ul style="margin:4px 0 0 18px;list-style:none;padding:0;">{rows}</ul>'
        '</details>'
    )


def _pmid_cite(pmid: str) -> str:
    """Small ' [PMID 123]' PubMed link, or '' when there's no citation."""
    pmid = (pmid or "").strip()
    if not pmid:
        return ""
    return (f' <a href="https://pubmed.ncbi.nlm.nih.gov/{html.escape(pmid)}/" target="_blank" '
            f'rel="noopener" style="color:#0984E3;font-size:0.75rem;">[PMID {html.escape(pmid)}]</a>')


def _mechanism_summary_html(summary: dict) -> str:
    """Collapsible 'Mechanism summary' block: compound, target, animal results, repurposed-from.

    RAG-grounded (offline step 5.5). Each field shows its supporting PMID when the claim came
    from the corpus; missing/unsupported fields read "Unknown", per the grounding guardrail.
    """
    if not summary:
        return ""

    def _val(text: str) -> str:
        text = (text or "unknown").strip()
        style = "color:#aaa;" if text.lower() in ("unknown", "not repurposed") else ""
        return f'<span style="{style}">{html.escape(text)}</span>'

    rows = [
        ("Compound", _val(summary.get("compound", "unknown")), ""),
        ("Targeting mechanism", _val(summary.get("targeting_mechanism", "unknown")),
         summary.get("targeting_mechanism_pmid", "")),
        ("Animal / preclinical results", _val(summary.get("animal_results", "unknown")),
         summary.get("animal_results_pmid", "")),
        ("Repurposed from", _val(summary.get("repurposed_from", "unknown")),
         summary.get("repurposed_from_pmid", "")),
    ]
    items = "".join(
        f'<li style="margin:2px 0;"><b>{label}:</b> {value}{_pmid_cite(pmid)}</li>'
        for label, value, pmid in rows
    )
    return (
        '<details style="margin-top:6px;font-size:0.82rem;color:#555;">'
        '<summary style="cursor:pointer;color:#6C5CE7;">Mechanism summary</summary>'
        f'<ul style="margin:4px 0 0 18px;list-style:none;padding:0;line-height:1.45;">{items}</ul>'
        '</details>'
    )


def _eligibility_html(elig: dict) -> str:
    """Collapsible enrollment-criteria block: age/sex summary + inclusion/exclusion text.

    Rendered only for active/recruiting trials (the caller gates on status), since that's when
    a physician assesses whether a patient qualifies.
    """
    criteria = (elig.get("criteria") or "").strip()
    if not criteria:
        return ""
    bits = []
    age = " – ".join(x for x in (elig.get("min_age"), elig.get("max_age")) if x) or None
    if age:
        bits.append(f"Age {html.escape(age)}")
    sex = elig.get("sex")
    if sex and sex != "ALL":
        bits.append(html.escape(sex.title()))
    elif sex == "ALL":
        bits.append("All sexes")
    if elig.get("healthy_volunteers"):
        bits.append("Accepts healthy volunteers")
    summary = "Eligibility" + (f" · {' · '.join(bits)}" if bits else "")
    body = html.escape(criteria).replace("\n", "<br>")
    return (
        '<details style="margin-top:6px;font-size:0.82rem;color:#555;">'
        f'<summary style="cursor:pointer;color:#0984E3;">{summary}</summary>'
        f'<div style="margin:4px 0 0 4px;line-height:1.4;">{body}</div>'
        '</details>'
    )


def render_trials_html(enriched: list[dict], match_count: int) -> str:
    """Render enriched location-search results as an HTML card list."""
    if not enriched:
        return '<div style="color:#888;padding:12px 0;">No trials found for this location.</div>'

    caption = (f'<div style="font-size:0.85rem;color:#666;margin:6px 0;">'
               f'{match_count} trial(s) matched — showing {len(enriched)}, recruiting first.</div>')

    cards = []
    for t in enriched:
        s_color, s_label = _status_badge(t["status"])
        ev = t["evidence"]
        tier_pill = _pill(f'Evidence: {ev["tier"]}', _TIER_COLOR.get(ev["tier"], "#B2BEC3"))
        mech = t.get("mechanism", "")
        mech_pill = _pill(f'Mechanism: {mech}', "#6C5CE7") if mech else ""
        rationale_html = _tier_rationale_html(ev)
        mech_summary_html = _mechanism_summary_html(t.get("mechanism_summary", {}))
        # Enrollment criteria only for active/recruiting trials — the enrollable ones.
        elig_html = _eligibility_html(t.get("eligibility", {})) if t.get("is_recruiting") else ""
        phase = html.escape((t.get("phase") or "—").replace("PHASE", "Ph"))
        title = html.escape(t.get("title", "")[:140])
        nct = html.escape(t.get("nct_id", ""))
        url = html.escape(t.get("url", ""))
        nct_link = f'<a href="{url}" target="_blank" rel="noopener">{nct}</a>' if url else nct

        sites = t.get("matched_sites", [])
        site_bits = []
        for st in sites[:3]:
            loc = ", ".join(x for x in (st.get("facility", ""), st.get("city", ""), st.get("state", "")) if x)
            if loc:
                site_bits.append(html.escape(loc))
        sites_html = "<br>".join(site_bits)
        if len(sites) > 3:
            sites_html += f'<br><span style="color:#aaa;">+{len(sites) - 3} more site(s)</span>'

        papers = t.get("key_papers", [])
        if papers:
            paper_items = "".join(
                f'<li>{html.escape(p.get("title", "")[:120])} '
                f'({p.get("year") or "n.d."}) — '
                f'<a href="https://pubmed.ncbi.nlm.nih.gov/{html.escape(str(p.get("pmid", "")))}/" '
                f'target="_blank" rel="noopener">PMID {html.escape(str(p.get("pmid", "")))}</a>, '
                f'{p.get("citation_count", 0)} citations</li>'
                for p in papers
            )
            papers_html = f'<div style="margin-top:6px;font-size:0.82rem;color:#555;"><b>Key papers:</b><ul style="margin:2px 0 0 18px;">{paper_items}</ul></div>'
        else:
            papers_html = '<div style="margin-top:6px;font-size:0.82rem;color:#aaa;">No supporting papers found in the database.</div>'

        siblings = t.get("sibling_trials", [])
        if siblings:
            sib_links = ", ".join(
                f'<a href="{html.escape(s.get("url", ""))}" target="_blank" rel="noopener">{html.escape(s.get("nct_id", ""))}</a>'
                for s in siblings
            )
            siblings_html = f'<div style="margin-top:4px;font-size:0.82rem;color:#555;"><b>Related trials (same compound):</b> {sib_links}</div>'
        else:
            siblings_html = ""

        cards.append(
            '<div style="border:1px solid #e3e3e3;border-radius:8px;padding:10px 12px;margin:8px 0;">'
            f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:4px;">'
            f'{_pill(s_label, s_color)}{tier_pill}{mech_pill}'
            f'<span style="color:#888;font-size:0.78rem;">{phase}</span></div>'
            f'<div style="font-weight:600;">{nct_link} — {title}</div>'
            f'{rationale_html}{mech_summary_html}'
            f'<div style="margin-top:4px;font-size:0.82rem;color:#555;"><b>Site(s):</b><br>{sites_html}</div>'
            f'{elig_html}{papers_html}{siblings_html}'
            '</div>'
        )

    return caption + "".join(cards)
