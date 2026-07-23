"""
Query-time drug-name canonicalization: variance + typo auto-fix.

Behavior:
  1. Variance (hyphen/space/case) — auto-handled downstream by normalized matching, e.g.
     'Prime-C'/'PrimeC'/'prime c' all match 'primec'. No note.
  2. Alias (brand↔generic, code↔name) — resolved via extraction.normalizer tables. No note.
  3. Typo — NOT auto-substituted. A conservative fuzzy match (rapidfuzz) yields a transparent
     "did you mean X?" SUGGESTION only; the original term is still what gets searched.

Why typos are suggested, not silently fixed: biomedical names are adversarially dense. Empirically
'biib068'→'BIIB078' (a DIFFERENT drug) scores 85.7, higher than the genuine typo 'primce'→'primec'
at 83.3 — so no score threshold can auto-correct the real typo without also silently mapping a
query onto the wrong drug. For a physician tool that is unacceptable, so tier 3 never changes the
searched term; it only surfaces a suggestion the physician can choose to act on.
"""
from __future__ import annotations

from rapidfuzz import fuzz, process

from extraction.normalizer import _COMPOUND_ALIASES, _GENE_ALIASES
from logging_config import get_logger

_logger = get_logger("normalization.drug_vocab")

# Fuzzy-SUGGESTION guards (calibrated: primce/primec ≈ 83; word-drug typos ≈ 87–94).
# Suggestions never change the searched term, so no ambiguity gap is needed — the physician
# sees the candidate and decides. Length floor still skips short gene codes (SOD1, NEK1, FUS).
_FUZZY_MIN_LEN = 6
_FUZZY_MIN_SCORE = 82.0


def _norm(s: str) -> str:
    """Lowercase, alphanumeric-only form so 'Prime-C'/'PrimeC'/'prime c' unify to 'primec'."""
    return "".join(c for c in s.lower() if c.isalnum())


def build_drug_vocab(trials: list[dict], graph=None) -> dict[str, object]:
    """
    Build the query-time drug vocabulary as {"exact": frozenset, "fuzzy": {norm: display}}.

    - `exact` (tier 1/2 membership): every normalized known name — alias forms, trial
      intervention names + target_entities, and KG Compound names. Broad on purpose so a real
      drug/intervention always resolves exactly and is NEVER sent to fuzzy correction.
    - `fuzzy` (tier 3 candidates): clean CANONICAL drug names only — alias-table values and KG
      Compound display names. Excludes messy intervention strings ("riluzole 50 mg tablet")
      whose near-duplicates would otherwise act as false ambiguity competitors and block valid
      corrections. Built once at startup and passed into the search handler.
    """
    exact: set[str] = set()
    fuzzy: dict[str, str] = {}

    def _add_exact(name: str) -> None:
        n = _norm(name)
        if len(n) >= 3:
            exact.add(n)

    def _add_fuzzy(name: str) -> None:
        n = _norm(name)
        if len(n) >= 3:
            fuzzy.setdefault(n, name)
            exact.add(n)

    # Alias tables — canonical values are clean drug names (fuzzy), surface forms exact-only.
    for alias, canonical in {**_COMPOUND_ALIASES, **_GENE_ALIASES}.items():
        _add_exact(alias)
        _add_fuzzy(canonical)

    # KG Compound node display names — clean canonical drug names (fuzzy candidates).
    if graph is not None:
        for _, data in graph.nodes(data=True):
            if data.get("type") == "Compound":
                _add_fuzzy(data.get("display_name", ""))

    # Trial interventions: exact for all; also fuzzy for clean single-token names
    # (e.g. "PrimeC", "AMX0035") but NOT verbose dosing strings ("riluzole 50 mg tablet").
    # Heuristic: no spaces → clean drug code/name → eligible for fuzzy suggestion.
    for t in trials or []:
        for iv in t.get("interventions", []):
            name = iv.get("name", "")
            _add_exact(name)
            if " " not in name.strip():
                _add_fuzzy(name)
        for tgt in t.get("target_entities", []):
            _add_exact(tgt)

    _logger.info("drug vocabulary built", extra={"data": {"exact": len(exact), "fuzzy": len(fuzzy)}})
    return {"exact": frozenset(exact), "fuzzy": fuzzy}


def suggest_drug_term(term: str, vocab: dict[str, object]) -> str | None:
    """
    Return a "did you mean 'X'?" suggestion for an unrecognized query drug term, or None.

    NEVER substitutes — the caller keeps searching the original term. Suggestion fires only for
    unknown terms (absent from the exact vocabulary), of sufficient length, whose closest clean
    canonical drug name scores above the floor. Safe by construction: a wrong suggestion cannot
    silently redirect the search onto a different drug; the physician decides.
    """
    if not term or not term.strip():
        return None
    exact: frozenset = vocab.get("exact", frozenset())  # type: ignore[assignment]
    fuzzy: dict[str, str] = vocab.get("fuzzy", {})       # type: ignore[assignment]

    n = _norm(term)
    if len(n) < _FUZZY_MIN_LEN or n in exact or not fuzzy:
        return None

    match = process.extractOne(n, list(fuzzy.keys()), scorer=fuzz.ratio)
    if not match:
        return None
    best_key, best_score, _ = match
    if best_score >= _FUZZY_MIN_SCORE and _norm(fuzzy[best_key]) != n:
        resolved = fuzzy[best_key]
        _logger.info("drug typo suggestion", extra={"data": {
            "query_term": term, "suggested": resolved, "score": round(best_score, 1),
        }})
        return resolved
    return None
