"""Tier-2 LLM-as-judge.

Scores a synthesized answer against the evidence the agent actually retrieved.
Uses a pinned judge model (config.JUDGE_MODEL) that is deliberately different from
and stronger than the synthesis model, so we never grade an answer with the model
that wrote it. Structured output via a forced strict-schema tool call — the judge
returns a per-claim breakdown we reduce into groundedness / attribution / relevance.

The judge is non-deterministic and costs tokens; it runs nightly / on-demand, never
in the PR gate. See docs/eval-plan.md §4 (Tier 2).
"""
from __future__ import annotations

import json

import anthropic

from config import JUDGE_MODEL

_JUDGE_TOOL = {
    "name": "report_judgment",
    "description": "Report the structured evaluation of the answer against the evidence.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["claims", "answer_relevance", "abstained", "completeness"],
        "properties": {
            "claims": {
                "type": "array",
                "description": "Every distinct factual or hedging claim the answer makes.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "is_factual", "grounded", "cited_pmids", "citations_support"],
                    "properties": {
                        "text": {"type": "string", "description": "The claim, quoted or paraphrased."},
                        "is_factual": {
                            "type": "boolean",
                            "description": "True if this is a substantive factual/medical claim that requires a citation. False for hedges, framing, or 'no evidence found' statements.",
                        },
                        "grounded": {
                            "type": "boolean",
                            "description": "True if the claim is supported by at least one of the evidence excerpts provided.",
                        },
                        "cited_pmids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "PMIDs the answer attaches to THIS claim (empty if none).",
                        },
                        "citations_support": {
                            "type": "boolean",
                            "description": "True if every PMID in cited_pmids genuinely supports this claim per its excerpt. True by convention when cited_pmids is empty.",
                        },
                    },
                },
            },
            "answer_relevance": {
                "type": "number",
                "description": "0.0–1.0: how well the answer addresses the physician's actual question.",
            },
            "abstained": {
                "type": "boolean",
                "description": "True if the answer declines / states there is insufficient evidence rather than asserting a mechanism.",
            },
            "completeness": {
                "type": "number",
                "description": "0.0–1.0 coverage vs the reference answer, or -1 if no reference was provided.",
            },
        },
    },
}

_SYSTEM = """You are a meticulous evaluator of a physician-facing ALS research assistant.
You are given a physician's QUESTION, the assistant's ANSWER, and the EVIDENCE excerpts
the assistant retrieved (each with a PMID). Judge ONLY against the provided evidence —
never your own knowledge.

Decompose the answer into distinct claims. For each claim decide:
- is_factual: does it assert a substantive medical/scientific fact that needs a source?
- grounded: is it supported by at least one evidence excerpt?
- cited_pmids: which PMIDs the answer attaches to that specific claim.
- citations_support: do those cited PMIDs actually support the claim per their excerpts?

Then score answer_relevance (does it answer the question), abstained (does it correctly
decline when evidence is thin), and completeness vs the reference answer if one is given
(else -1). Be strict: a real but wrong citation is NOT supporting. Report via the tool."""


def _evidence_block(evidence: list[dict]) -> str:
    if not evidence:
        return "(no evidence retrieved)"
    lines = []
    for e in evidence:
        pmid = e.get("pmid", "?")
        title = e.get("title", "")
        excerpt = (e.get("excerpt") or "").strip().replace("\n", " ")
        lines.append(f"[PMID {pmid}] {title}\n{excerpt}")
    return "\n\n".join(lines)


def judge_answer(
    question: str,
    answer: str,
    evidence: list[dict],
    reference_answer: str | None = None,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """Run the judge on one answer. Returns the raw report_judgment payload."""
    client = client or anthropic.Anthropic()

    user = (
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer}\n\n"
        f"EVIDENCE (only these excerpts may ground a claim):\n{_evidence_block(evidence)}\n\n"
        f"REFERENCE ANSWER:\n{reference_answer or '(none provided — set completeness to -1)'}"
    )

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=4096,
        system=_SYSTEM,
        tools=[_JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "report_judgment"},
        messages=[{"role": "user", "content": user}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "report_judgment":
            # Parse defensively — tool inputs may carry model-specific JSON escaping.
            return block.input if isinstance(block.input, dict) else json.loads(block.input)
    raise RuntimeError("judge did not return a report_judgment tool call")


def reduce_metrics(judgment: dict) -> dict:
    """Reduce a judge payload into scalar metrics."""
    claims = judgment.get("claims", [])
    factual = [c for c in claims if c.get("is_factual")]
    cited = [c for c in factual if c.get("cited_pmids")]

    groundedness = _safe_div(sum(c.get("grounded", False) for c in factual), len(factual))
    attribution_precision = _safe_div(
        sum(c.get("citations_support", False) for c in cited), len(cited))
    attribution_recall = _safe_div(len(cited), len(factual))

    completeness = judgment.get("completeness", -1)
    return {
        "groundedness": groundedness,
        "attribution_precision": attribution_precision,
        "attribution_recall": attribution_recall,
        "answer_relevance": float(judgment.get("answer_relevance", 0.0)),
        "abstained": bool(judgment.get("abstained", False)),
        "completeness": (float(completeness) if completeness is not None and completeness >= 0 else None),
        "n_factual_claims": len(factual),
        "n_cited_claims": len(cited),
    }


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 1.0
