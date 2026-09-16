"""Offline: validate gold files parse and carry the required fields.

Cheap sanity so a malformed gold edit fails fast in PR CI rather than blowing up
a downstream check with a confusing traceback.
"""
from __future__ import annotations

import json

from evals import thresholds as T
from evals.checks.result import CheckResult

_ALIAS_REQUIRED = {"group", "entity_type", "canonical_id", "surface_forms"}
_QUESTION_REQUIRED = {"id", "question", "query_entities", "expected_entities", "relevant_pmids"}


def _load_jsonl(path) -> tuple[list[dict], list[str]]:
    rows, errs = [], []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            errs.append(f"{path.name}:{i}: invalid JSON — {e}")
    return rows, errs


def run() -> CheckResult:
    details: list[str] = []

    if not T.ALIASES_GOLD.exists():
        details.append(f"missing {T.ALIASES_GOLD}")
    if not T.QUESTIONS_GOLD.exists():
        details.append(f"missing {T.QUESTIONS_GOLD}")
    if details:
        return CheckResult("gold-schema", "offline", "fail",
                           "gold file(s) missing", details=details)

    aliases, aerr = _load_jsonl(T.ALIASES_GOLD)
    questions, qerr = _load_jsonl(T.QUESTIONS_GOLD)
    details += aerr + qerr

    for r in aliases:
        missing = _ALIAS_REQUIRED - r.keys()
        if missing:
            details.append(f"alias '{r.get('group', '?')}' missing fields {sorted(missing)}")
        elif len(r["surface_forms"]) < 2:
            details.append(f"alias '{r['group']}' needs >=2 surface_forms to test collapse")

    ids = set()
    for r in questions:
        missing = _QUESTION_REQUIRED - r.keys()
        if missing:
            details.append(f"question '{r.get('id', '?')}' missing fields {sorted(missing)}")
        if r.get("id") in ids:
            details.append(f"duplicate question id '{r['id']}'")
        ids.add(r.get("id"))

    status = "fail" if details else "pass"
    summary = (f"{len(aliases)} alias groups, {len(questions)} questions"
               if status == "pass" else f"{len(details)} schema problem(s)")
    return CheckResult("gold-schema", "offline", status, summary,
                       metrics={"alias_groups": len(aliases), "questions": len(questions)},
                       details=details)
