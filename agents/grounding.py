"""Post-generation grounding gate for synthesis output.

The synthesis prompt (``SYNTHESIS_SYSTEM``) instructs the model to attach an inline
``(PMID: N)`` to every mechanism / entity claim and to omit anything it cannot ground in a
retrieved excerpt. This module enforces that in code rather than trusting the model to
self-police: it strips any bullet in a gated section whose cited PMID's *retrieved excerpt*
does not actually support the claim (or that cites no PMID at all, or cites a PMID that was
never retrieved). A claim about a real-world mechanism grafted onto a topically-adjacent paper
is exactly what this removes.

The check is a deliberately conservative lexical-overlap heuristic — it can only see the
excerpts retrieval delivered (``document[:600]`` per paper), so it errs toward keeping a bullet
unless the overlap is clearly too low to be a genuine citation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Sections whose bullets must be grounded in a cited excerpt. Matched against the text of a
# ``## Header`` line, case-insensitively.
GATED_SECTIONS: tuple[str, ...] = ("key mechanisms", "entities involved")

# A bullet is considered grounded when at least this fraction of its distinctive claim terms
# appear in the concatenated excerpts of the PMIDs it cites. Conservative on purpose: low
# enough that honest paraphrase survives, high enough that an unsupported claim is dropped.
MIN_OVERLAP = 0.4

_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_BULLET_RE = re.compile(r"^\s*[-*]\s+\S")
_PMID_RE = re.compile(r"PMID:\s*(\d+)", re.IGNORECASE)
_CITATION_RE = re.compile(r"\(\s*PMID:[^)]*\)", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9-]{2,}")

# Common words that carry no grounding signal. Kept small so distinctive claim terms
# (gene/compound/mechanism names) dominate the overlap ratio.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "its", "for", "with", "that", "this", "from", "into", "are", "was",
    "were", "has", "have", "had", "been", "their", "which", "also", "than", "then",
    "such", "these", "those", "via", "per", "not", "but", "can", "may", "could",
    "would", "should", "within", "between", "through", "including", "including",
    "role", "roles", "involved", "relevant", "query", "primary", "described",
    "mention", "summary", "table", "study", "studies", "paper", "papers", "context",
    "evidence", "clinical", "als", "amyotrophic", "lateral", "sclerosis",
})


@dataclass
class GroundingResult:
    """Outcome of applying the grounding gate to a synthesis answer."""

    text: str
    removed: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.removed)


def _content_words(text: str) -> set[str]:
    """Distinctive lowercase tokens, minus stopwords and bare numbers (e.g. stray years/PMIDs)."""
    return {
        w for w in _WORD_RE.findall(text.lower())
        if w not in _STOPWORDS and not w.isdigit()
    }


def _bullet_grounded(bullet: str, pmid_excerpts: dict[str, str]) -> bool:
    """True when the bullet's claim is supported by the excerpt(s) of the PMID(s) it cites.

    Returns False when the bullet cites no PMID, cites only PMIDs that were never retrieved,
    or when its claim terms overlap the cited excerpts below ``MIN_OVERLAP``.
    """
    cited = _PMID_RE.findall(bullet)
    if not cited:
        return False

    excerpts = " ".join(pmid_excerpts.get(pmid, "") for pmid in cited)
    if not excerpts.strip():
        # Every cited PMID is absent from what retrieval actually returned.
        return False

    claim = _CITATION_RE.sub(" ", bullet)  # drop the citation parenthetical before scoring
    claim_words = _content_words(claim)
    if not claim_words:
        # Nothing substantive to verify (e.g. a lone cross-reference); leave it alone.
        return True

    excerpt_words = _content_words(excerpts)
    overlap = len(claim_words & excerpt_words) / len(claim_words)
    return overlap >= MIN_OVERLAP


def _withheld_note(n: int) -> str:
    claim = "claim" if n == 1 else "claims"
    return (
        f"*(This tool withheld {n} {claim} here: no retrieved excerpt supported the "
        f"statement, so it was removed rather than shown uncited.)*"
    )


def enforce_grounding(answer: str, pmid_excerpts: dict[str, str]) -> GroundingResult:
    """Strip ungrounded bullets from the gated sections of a synthesis answer.

    Args:
        answer: the model's full markdown answer.
        pmid_excerpts: PMID (as ``str``) → the retrieved excerpt text the model was given.

    Returns:
        A ``GroundingResult`` with the cleaned ``text`` and the list of ``removed`` bullet lines.
        When a gated section loses one or more bullets, a short italic note is inserted so the
        omission is visible to the physician.
    """
    pmid_excerpts = {str(k): v for k, v in pmid_excerpts.items()}

    out: list[str] = []
    removed: list[str] = []
    in_gated = False
    removed_in_section = 0

    def _flush_section_note() -> None:
        nonlocal removed_in_section
        if removed_in_section:
            out.append(_withheld_note(removed_in_section))
            removed_in_section = 0

    for line in answer.splitlines():
        header = _HEADER_RE.match(line)
        if header:
            # A new header closes any gated section we were inside.
            _flush_section_note()
            in_gated = header.group(1).strip().lower() in GATED_SECTIONS
            out.append(line)
            continue

        if in_gated and _BULLET_RE.match(line) and not _bullet_grounded(line, pmid_excerpts):
            removed.append(line.strip())
            removed_in_section += 1
            continue

        out.append(line)

    _flush_section_note()

    text = "\n".join(out)
    if answer.endswith("\n") and not text.endswith("\n"):
        text += "\n"
    return GroundingResult(text=text, removed=removed)
