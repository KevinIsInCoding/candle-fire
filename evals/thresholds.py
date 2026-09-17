"""All eval-gate thresholds in one place.

Set at or just below the current baseline so the gate catches *regressions*
without blocking on day one. Re-baseline deliberately when the pipeline
genuinely improves — a threshold bump is a reviewable line in a PR.
"""
from __future__ import annotations

from pathlib import Path

EVALS_DIR = Path(__file__).parent
GOLD_DIR = EVALS_DIR / "gold"
REPORT_DIR = EVALS_DIR / "report"

ALIASES_GOLD = GOLD_DIR / "aliases.jsonl"
QUESTIONS_GOLD = GOLD_DIR / "questions.jsonl"

# ── Normalization (offline) ───────────────────────────────────────────────────
# Alias-collapse invariant: two surface forms of the same entity must resolve to
# one canonical_id. Zero tolerance — this is a hard project invariant.
NORMALIZATION_MIN_ACCURACY = 1.0

# ── Therapy landscape classifier (offline) ────────────────────────────────────
# Baseline (2026-09-16): precision 0.90, F1 0.80. Physician's rule — a wrong
# label is worse than a missing one — so precision is the tighter gate.
LANDSCAPE_MIN_PRECISION = 0.85
LANDSCAPE_MIN_F1 = 0.75

# ── KG expansion (data) ───────────────────────────────────────────────────────
# Fraction of a question's expected entities that must appear (case-insensitive
# substring) in the expanded set. Recall-style: expansion may add more, but must
# not drop the expected neighbors.
KG_EXPANSION_MIN_RECALL = 0.80

# ── Retrieval (data) ──────────────────────────────────────────────────────────
# Binary relevance (option C — graded/nDCG deferred). Recall@k is the gate;
# precision@k and MRR are reported for trend but not gated yet (small gold set).
RETRIEVAL_K = 10
RETRIEVAL_MIN_RECALL_AT_K = 0.70

# ── Citation existence (data) ─────────────────────────────────────────────────
# Every reference/citation PMID in the gold set must exist in the corpus.
# Zero tolerance — a cited PMID the corpus can't produce is a fabrication risk.
CITATION_MIN_VALIDITY = 1.0

# ── Tier-2 LLM judge (non-blocking, nightly) ──────────────────────────────────
# These are ALERT thresholds, not PR gates: the judge is non-deterministic and
# costs tokens, so it never runs in PR CI. A nightly run below these turns red to
# notify, but no PR is ever blocked. Re-baseline once the judge is calibrated
# against SME grades (see docs/eval-plan.md §7 judge calibration).
JUDGE_MIN_GROUNDEDNESS = 0.90
JUDGE_MIN_ATTRIBUTION_PRECISION = 0.90
JUDGE_MIN_ATTRIBUTION_RECALL = 0.80
JUDGE_MIN_ANSWER_RELEVANCE = 0.80
# Abstention + completeness are reported for trend only (no gate yet).
