"""Facility / geography trial search + research-evidence enrichment.

Physicians ask "what trials are at Mass General Hospital" or "what trials are in NY".
This module filters the offline-ingested trials list by facility/city/state/country
(in-memory — no fetch at query time) and enriches each hit with surrounding research
evidence: key papers, sibling trials for the same compound, a heuristic evidence-strength
tier, and recruiting status.
"""
from __future__ import annotations

import html
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
) -> list[dict]:
    """Return trials with at least one site matching the location filters.

    `status`: "Recruiting" keeps only trials with an open overall status; "Not recruiting"
    keeps only closed ones; anything else (None / "All") keeps all. Each returned trial is a
    shallow copy with `matched_sites` attached; recruiting trials are ranked first.
    """
    if not any([facility, city, state, country]):
        return []

    status_filter = (status or "").strip().lower()
    results: list[dict] = []
    for trial in trials:
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


def _kg_paper_count(trial: dict, graph: "nx.DiGraph | None") -> int:
    """Max supporting-paper count across this trial's target entities in the KG."""
    if graph is None:
        return 0
    from graph.query import _find_node

    best = 0
    for name in trial.get("target_entities", []):
        for node_id in _find_node(graph, name):
            best = max(best, graph.nodes[node_id].get("paper_count", 0))
    return best


def _evidence_tier(
    trial: dict,
    key_papers: list[dict],
    kg_paper_count: int,
) -> dict:
    """Heuristic evidence-strength tier from paper count, citations, KG breadth, and phase.

    Strong / Moderate / Emerging — a fully offline signal (no LLM call).
    """
    n_papers = len(key_papers)
    max_cit = max((p.get("citation_count", 0) for p in key_papers), default=0)
    phase = (trial.get("phase", "") or "").upper()

    points = 0
    if n_papers >= 5:
        points += 2
    elif n_papers >= 2:
        points += 1
    if max_cit >= 100:
        points += 2
    elif max_cit >= 20:
        points += 1
    if kg_paper_count >= 5:
        points += 1
    if "PHASE3" in phase.replace(" ", "") or "3" in phase:
        points += 1

    if points >= 4:
        tier = "Strong"
    elif points >= 2:
        tier = "Moderate"
    else:
        tier = "Emerging"

    return {
        "tier": tier,
        "n_papers": n_papers,
        "max_citations": max_cit,
        "kg_paper_count": kg_paper_count,
    }


def enrich_trial(
    trial: dict,
    collection: "chromadb.Collection | None" = None,
    graph: "nx.DiGraph | None" = None,
    all_trials: list[dict] | None = None,
) -> dict:
    """Attach research-evidence context to a trial: key papers, sibling trials, evidence tier."""
    key_papers = _find_supporting_papers(trial, collection)
    sibling_trials = _find_sibling_trials(trial, all_trials or [])
    kg_paper_count = _kg_paper_count(trial, graph)
    evidence = _evidence_tier(trial, key_papers, kg_paper_count)

    return {
        "nct_id": trial.get("nct_id", ""),
        "title": trial.get("title", ""),
        "phase": trial.get("phase", ""),
        "status": trial.get("status", ""),
        "is_recruiting": trial.get("status", "") in RECRUITING_STATUSES,
        "sponsor": trial.get("sponsor", ""),
        "url": trial.get("url", ""),
        "target_entities": trial.get("target_entities", []),
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
            f'{_pill(s_label, s_color)}{tier_pill}'
            f'<span style="color:#888;font-size:0.78rem;">{phase}</span></div>'
            f'<div style="font-weight:600;">{nct_link} — {title}</div>'
            f'<div style="margin-top:4px;font-size:0.82rem;color:#555;"><b>Site(s):</b><br>{sites_html}</div>'
            f'{papers_html}{siblings_html}'
            '</div>'
        )

    return caption + "".join(cards)
