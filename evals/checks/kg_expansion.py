"""Data: KG-expansion recall.

For each question, expand its query_entities through the graph and check that the
expected neighbor concepts appear in the expanded set (case-insensitive
substring). Recall-style — expansion may add more, but must not drop the expected
neighbors. Guards the "KG expansion precedes RAG" invariant against graph or
traversal regressions.
"""
from __future__ import annotations

from graph.query import expand_query_entities
from evals import thresholds as T
from evals.checks.context import DataContext
from evals.checks.result import CheckResult


def run(ctx: DataContext) -> CheckResult:
    questions = [q for q in ctx.questions if q.get("expected_entities")]
    if not questions:
        return CheckResult("kg-expansion", "data", "skip",
                           "no questions with expected_entities")

    per_q_recall: list[float] = []
    details: list[str] = []
    for q in questions:
        expanded = expand_query_entities(ctx.graph, q["query_entities"])
        blob = " || ".join(expanded).lower()
        expected = q["expected_entities"]
        hits = [e for e in expected if e.lower() in blob]
        recall = len(hits) / len(expected)
        per_q_recall.append(recall)
        if recall < T.KG_EXPANSION_MIN_RECALL:
            missing = [e for e in expected if e.lower() not in blob]
            details.append(f"{q['id']}: recall {recall:.2f}, missing {missing}")

    mean_recall = sum(per_q_recall) / len(per_q_recall)
    status = "fail" if details else "pass"
    return CheckResult(
        "kg-expansion", "data", status,
        f"mean expected-entity recall {mean_recall:.2f} over {len(questions)} questions",
        metrics={"mean_recall": round(mean_recall, 4), "questions": len(questions)},
        thresholds={"min_recall": T.KG_EXPANSION_MIN_RECALL},
        details=details,
    )
