"""Tests for the post-generation grounding gate (agents/grounding.py)."""
from __future__ import annotations

from agents.grounding import GATED_SECTIONS, enforce_grounding


def _mechanisms(*bullets: str) -> str:
    body = "\n".join(bullets)
    return f"## Key Mechanisms\n{body}\n\n## Evidence Strength\nSolid.\n"


def test_grafted_claim_is_removed():
    """A real-world claim cited to a paper whose excerpt does not mention it is stripped."""
    answer = _mechanisms(
        "- SPG302 targets a regulator of the F-actin-based cytoskeleton, promoting "
        "regeneration of glutamatergic synapses (PMID: 40858858)."
    )
    # Excerpt for the cited PMID is about something entirely different.
    excerpts = {"40858858": "TDP-43 proteinopathy drives motor neuron degeneration in ALS."}

    result = enforce_grounding(answer, excerpts)

    assert result.changed
    assert len(result.removed) == 1
    assert "F-actin" not in result.text
    assert "withheld 1 claim" in result.text
    # Untouched sections survive.
    assert "## Evidence Strength" in result.text
    assert "Solid." in result.text


def test_grounded_claim_is_kept():
    """A claim whose terms appear in the cited excerpt is preserved untouched."""
    answer = _mechanisms(
        "- SPG302 targets a regulator of the F-actin-based cytoskeleton, promoting "
        "regeneration of glutamatergic synapses (PMID: 40858858)."
    )
    excerpts = {
        "40858858": (
            "SPG302 and its related compounds target a regulator of the F-actin-based "
            "cytoskeleton, thereby promoting the regeneration of glutamatergic synapses."
        )
    }

    result = enforce_grounding(answer, excerpts)

    assert not result.changed
    assert result.removed == []
    assert "F-actin" in result.text
    assert "withheld" not in result.text


def test_uncited_bullet_is_removed():
    """A mechanism bullet with no PMID at all cannot be grounded and is dropped."""
    answer = _mechanisms("- Oxidative stress contributes to motor neuron death.")

    result = enforce_grounding(answer, {"999": "irrelevant excerpt"})

    assert result.changed
    assert "Oxidative stress" not in result.text
    assert "withheld 1 claim" in result.text


def test_citation_to_unretrieved_pmid_is_removed():
    """A bullet citing a PMID that retrieval never returned is dropped."""
    answer = _mechanisms(
        "- Glutamate excitotoxicity drives calcium influx and neuronal death (PMID: 12345678)."
    )
    # The cited PMID is absent from the retrieved excerpt map.
    result = enforce_grounding(answer, {"40858858": "unrelated text about SOD1"})

    assert result.changed
    assert "Glutamate excitotoxicity" not in result.text


def test_multiple_bullets_mixed():
    """Grounded bullets stay, ungrounded ones go, with a single withheld note per section."""
    answer = _mechanisms(
        "- SOD1 mutations cause toxic protein misfolding and aggregation (PMID: 111).",
        "- SPG302 regenerates glutamatergic synapses via F-actin regulation (PMID: 222).",
    )
    excerpts = {
        "111": "SOD1 mutations cause toxic protein misfolding and aggregation in motor neurons.",
        "222": "This review summarizes emerging ALS therapeutics and trial phases.",
    }

    result = enforce_grounding(answer, excerpts)

    assert len(result.removed) == 1
    assert "SOD1 mutations" in result.text
    assert "SPG302" not in result.text
    assert result.text.count("withheld") == 1


def test_only_gated_sections_are_touched():
    """Bullets outside the gated sections are never removed, even if uncited."""
    answer = (
        "## Key Citations\n"
        "- Some Paper (2024) — PMID: 999\n\n"
        "## Related Clinical Trials\n"
        "- NCT12345678 — Phase 2, recruiting\n"
    )
    result = enforce_grounding(answer, {})

    assert not result.changed
    assert "NCT12345678" in result.text
    assert "Some Paper" in result.text


def test_entities_section_is_gated():
    """The Entities Involved section is gated too."""
    assert "entities involved" in GATED_SECTIONS
    answer = (
        "## Entities Involved\n"
        "- FUS: an RNA-binding protein implicated in ALS aggregation (PMID: 555).\n"
    )
    result = enforce_grounding(answer, {"555": "Riluzole modulates glutamate release."})

    assert result.changed
    assert "FUS" not in result.text
