"""Tier-2 (judge): end-to-end synthesis quality.

Runs the real research agent on each gold question, captures the answer plus the
evidence it actually retrieved, and scores it with the pinned judge model. Emits
one CheckResult per metric — groundedness, attribution (precision+recall), answer
relevance, and (when the gold set has them) abstention and completeness.

Non-blocking by design: this suite never runs in PR CI. A nightly run below the
alert thresholds turns red to notify, but no PR is gated. See docs/eval-plan.md §4.
"""
from __future__ import annotations

import json

from evals import thresholds as T
from evals.checks.context import DataContext
from evals.checks.result import CheckResult
from evals.judge import judge_answer, reduce_metrics


def _run_agent(ctx: DataContext, client, question: str) -> tuple[str, list[dict]]:
    """Drive the agent to completion; return (final answer, retrieved evidence papers)."""
    from agents.research_agent import stream_research_agent

    answer = ""
    papers: dict[str, dict] = {}  # dedup by pmid across multiple searches
    for event_type, content in stream_research_agent(
        client, question, ctx.collection, ctx.trials, graph=ctx.graph
    ):
        if event_type == "done":
            answer = content
        elif event_type == "evidence":
            for p in json.loads(content).get("papers", []):
                papers[p["pmid"]] = p
    return answer, list(papers.values())


def run(ctx: DataContext) -> list[CheckResult]:
    import anthropic

    questions = ctx.questions
    client = anthropic.Anthropic()

    per_q: list[dict] = []
    for q in questions:
        answer, evidence = _run_agent(ctx, client, q["question"])
        reference = (q.get("reference_answers") or [None])[0]
        judgment = judge_answer(q["question"], answer, evidence, reference_answer=reference, client=client)
        m = reduce_metrics(judgment)
        m["id"] = q["id"]
        m["adversarial"] = bool(q.get("adversarial"))
        per_q.append(m)

    return _build_results(per_q)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _build_results(per_q: list[dict]) -> list[CheckResult]:
    results: list[CheckResult] = []

    def gate(name, values, threshold, label):
        mean = _mean(values)
        details = [f"{m['id']}: {m[name]:.2f}" for m in per_q if m[name] < threshold]
        status = "fail" if mean < threshold else "pass"
        return CheckResult(
            f"judge-{label}", "data", status,
            f"mean {label} {mean:.2f} over {len(values)} questions",
            metrics={f"mean_{name}": round(mean, 4),
                     "per_question": {m["id"]: round(m[name], 4) for m in per_q}},
            thresholds={f"min_{name}": threshold},
            details=details,
        )

    results.append(gate("groundedness", [m["groundedness"] for m in per_q],
                        T.JUDGE_MIN_GROUNDEDNESS, "groundedness"))

    # Attribution precision + recall as one paired check (the headline cross-reference metric).
    prec = [m["attribution_precision"] for m in per_q]
    rec = [m["attribution_recall"] for m in per_q]
    mp, mr = _mean(prec), _mean(rec)
    att_details = [f"{m['id']}: P={m['attribution_precision']:.2f} R={m['attribution_recall']:.2f}"
                   for m in per_q
                   if m["attribution_precision"] < T.JUDGE_MIN_ATTRIBUTION_PRECISION
                   or m["attribution_recall"] < T.JUDGE_MIN_ATTRIBUTION_RECALL]
    att_status = "fail" if (mp < T.JUDGE_MIN_ATTRIBUTION_PRECISION
                            or mr < T.JUDGE_MIN_ATTRIBUTION_RECALL) else "pass"
    results.append(CheckResult(
        "judge-attribution", "data", att_status,
        f"precision {mp:.2f} / recall {mr:.2f} over {len(per_q)} questions",
        metrics={"mean_precision": round(mp, 4), "mean_recall": round(mr, 4),
                 "per_question": {m["id"]: {"P": round(m["attribution_precision"], 4),
                                            "R": round(m["attribution_recall"], 4)} for m in per_q}},
        thresholds={"min_precision": T.JUDGE_MIN_ATTRIBUTION_PRECISION,
                    "min_recall": T.JUDGE_MIN_ATTRIBUTION_RECALL},
        details=att_details,
    ))

    results.append(gate("answer_relevance", [m["answer_relevance"] for m in per_q],
                        T.JUDGE_MIN_ANSWER_RELEVANCE, "relevance"))

    # Abstention — only over adversarial questions; correct == abstained.
    adv = [m for m in per_q if m["adversarial"]]
    if adv:
        correct = sum(m["abstained"] for m in adv)
        rate = correct / len(adv)
        results.append(CheckResult(
            "judge-abstention", "data", "pass",  # reported, not gated yet
            f"abstained correctly on {correct}/{len(adv)} adversarial questions",
            metrics={"abstention_rate": round(rate, 4), "adversarial": len(adv)},
        ))

    # Completeness — only when references exist; reported, not gated.
    comp = [m["completeness"] for m in per_q if m["completeness"] is not None]
    if comp:
        results.append(CheckResult(
            "judge-completeness", "data", "pass",
            f"mean completeness {_mean(comp):.2f} over {len(comp)} questions",
            metrics={"mean_completeness": round(_mean(comp), 4)},
        ))

    return results
