"""Data: retrieval recall@k / precision@k / MRR (binary relevance).

Runs semantic search per question and scores the ranked PMIDs against the gold
relevant_pmids. Binary labels (option C — graded/nDCG deferred). recall@k is the
gate; precision@k and MRR are reported for trend but not gated yet (small gold
set makes them noisy). Questions with no relevant_pmids are skipped, so the gold
set can grow incrementally without breaking the gate.
"""
from __future__ import annotations

from rag.retriever import search
from evals import thresholds as T
from evals.checks.result import CheckResult
from evals.checks.context import DataContext


def _metrics(ranked: list[str], relevant: set[str], k: int) -> tuple[float, float, float]:
    topk = ranked[:k]
    hits = [p for p in topk if p in relevant]
    recall = len(set(hits)) / len(relevant) if relevant else 0.0
    precision = len(hits) / k
    rr = 0.0
    for i, p in enumerate(ranked, 1):
        if p in relevant:
            rr = 1.0 / i
            break
    return recall, precision, rr


def run(ctx: DataContext) -> CheckResult:
    questions = [q for q in ctx.questions if q.get("relevant_pmids")]
    if not questions:
        return CheckResult("retrieval", "data", "skip",
                           "no questions with labeled relevant_pmids (pending labeling)")

    k = T.RETRIEVAL_K
    recalls, precisions, rrs = [], [], []
    details: list[str] = []
    for q in questions:
        relevant = {str(p) for p in q["relevant_pmids"]}
        ranked = [str(r["pmid"]) for r in search(ctx.collection, q["question"], n_results=max(k, 20))]
        recall, precision, rr = _metrics(ranked, relevant, k)
        recalls.append(recall)
        precisions.append(precision)
        rrs.append(rr)
        if recall < T.RETRIEVAL_MIN_RECALL_AT_K:
            found = [p for p in ranked[:k] if p in relevant]
            details.append(f"{q['id']}: recall@{k} {recall:.2f}, found {found} of {sorted(relevant)}")

    mean = lambda xs: sum(xs) / len(xs)
    status = "fail" if details else "pass"
    return CheckResult(
        "retrieval", "data", status,
        f"recall@{k}={mean(recalls):.2f} precision@{k}={mean(precisions):.2f} "
        f"MRR={mean(rrs):.2f} over {len(questions)} questions",
        metrics={f"recall_at_{k}": round(mean(recalls), 4),
                 f"precision_at_{k}": round(mean(precisions), 4),
                 "mrr": round(mean(rrs), 4), "questions": len(questions)},
        thresholds={f"min_recall_at_{k}": T.RETRIEVAL_MIN_RECALL_AT_K},
        details=details,
    )
