# Candle-Fire Evaluation Plan

**Document type:** Design / plan · **Author: Kevin Chen 
**Status:** Draft · **Date:** 2026-09-14
**Service:** candle-fire — physician-facing ALS research intelligence tool

---

## 1. Summary

Candle-fire answers free-text physician questions with a synthesized, cited
answer grounded in a curated ALS corpus (KG expansion → RAG retrieval → Claude
synthesis). Because the output is physician-facing and citation-bearing, the
worst failure is not a mediocre answer — it is a **confident, fabricated, or
misattributed citation**. Today we have no automated guard against that, and no
way to know whether a prompt, model, or re-index change has silently regressed
retrieval or grounding.

This plan proposes an evaluation harness with two purposes in priority order:
1. A **CI regression gate** that blocks changes which break grounding or degrade
   retrieval — fast, deterministic, no LLM calls, runs on every PR.
2. A **nightly quality track** that scores synthesis faithfulness and end-to-end
   answer quality with a pinned LLM judge — trended, not blocking.

Ground-truth labels are **LLM-bootstrapped and human spot-checked**, frozen as
versioned JSONL. The harness scores each pipeline layer independently and
reports per-layer scores so a regression points to *where* it broke.

**Scope of this plan:** retrieval + KG expansion, entity normalization,
synthesis faithfulness + citations, the therapy landscape classifier, and an
end-to-end holistic score.

---

## 2. Goals and Non-Goals

**Goals.**
1. Catch fabricated / misattributed citations on **every** commit, deterministically.
2. Detect retrieval and KG-expansion regressions before they reach production.
3. Give per-layer diagnostics so a score drop is actionable.
4. Score synthesis faithfulness and end-to-end quality on a cadence, with trend tracking.
5. Run entirely offline against committed assets (graph + ChromaDB) — no live PubMed/Entrez calls at eval time.

**Non-Goals (this phase).**
- Replacing physician/SME judgment for release sign-off. Human review stays a
  separate, low-frequency gate; this harness informs it, not replaces it.
- Latency / cost benchmarking (worth doing, out of scope here).
- Evaluating ingestion correctness (PubMed/PMC fetch fidelity) — assumed upstream.

---

## 3. Failure Modes, Ranked

Eval weighting follows this ranking. Catastrophic failures are hard gates;
quality failures are trended thresholds.

| Rank | Failure mode | Severity | Gate type |
|---|---|---|---|
| 1 | Fabricated citation — cited PMID not in corpus | Catastrophic | Hard (Tier 1) |
| 2 | Misattributed citation — PMID real but not retrieved / not supporting the claim | Catastrophic | Hard (Tier 1) + judge (Tier 2) |
| 3 | Ungrounded claim — assertion with no supporting retrieved passage | High | Judge (Tier 2) |
| 4 | Missed evidence — relevant corpus paper not retrieved | High | Threshold (Tier 1) |
| 5 | Entity collapse failure — `TDP-43` and `TARDBP` produce two nodes | Medium | Threshold (Tier 1) |
| 6 | Weak KG expansion — query entity fails to reach expected neighbors | Medium | Threshold (Tier 1) |
| 7 | Therapy misclassification vs. gold | Medium | Threshold (Tier 1) |
| 8 | Incomplete / low-relevance answer | Low–Medium | Judge (Tier 2) |

---

## 4. Two-Tier Design

CI must be fast and deterministic; faithfulness scoring needs an LLM judge that
is slow, costs tokens, and is non-deterministic. We split accordingly.

### Tier 1 — Blocking, every PR (deterministic, no LLM calls)
Runs in seconds against the committed graph + ChromaDB. Any Tier-1 failure
fails the build.

- **Citation validity gate** *(zero tolerance)* — for every cited PMID in every
  gold answer replay: (a) PMID exists in the corpus, and (b) PMID appeared in
  the retrieved set for that query. Uses `rag/retriever.py:get_paper` and the
  retrieval result set. Any violation fails the build.
- **Retrieval metrics** — recall@k, MRR, nDCG against gold relevant-PMID sets,
  via `search` / `search_by_entities`. Measures the effect of
  `apply_citation_boost` and `cross_encoder_rerank` re-ranking.
- **KG expansion overlap** — recall / Jaccard of `expand_query_entities` output
  vs. expected entity sets.
- **Entity normalization** — `normalize_entity` on gold alias pairs; asserts the
  canonical_id-collapse invariant.
- **Therapy landscape classifier** — accuracy + macro-F1 vs.
  `data/seeds/therapy_gold.json` (self-contained; first to build).

### Tier 2 — Nightly / on-label (LLM judge, non-blocking)
Runs on a schedule or an `eval` label. Reports scores + trend; alerts on drop
past a threshold but does not block routine PRs.

- **Synthesis faithfulness** — each claim in the answer is grounded in a
  retrieved passage. Pinned judge model.
- **Answer relevance + completeness** — vs. reference answers.
- **End-to-end holistic score** — full pipeline on the gold question set.

---

## 5. Metric Definitions (reference)

Grouped by which leg of the RAG triad they measure: retrieval (Context↔Query),
grounding/attribution (Answer↔Context), answer quality (Answer↔Query), and
safety.

### A. Retrieval (Context ↔ Query)

Retrieval is scored as a ranking problem: given a query, return a ranked list of
passages; the gold set marks which are relevant.

- **Recall@k** — of all relevant docs in the corpus, the fraction that appear in
  the top-k. `(relevant in top-k) / (total relevant)`. Catches **missed
  evidence**; it is the ceiling on everything downstream — if retrieval misses a
  paper, synthesis can never cite it.
- **Precision@k** — of the k returned, the fraction that are relevant.
  `(relevant in top-k) / k`. Catches **noise** in the synthesis context. Raising
  k trades precision for recall — the core tuning knob.
- **MRR** — mean of `1 / (rank of first relevant result)`. Measures how high the
  *first* good hit sits; ignores the rest.
- **nDCG** — rewards placing more-relevant docs higher, with a log discount by
  position, normalized against the ideal ordering (0–1). The richest single
  retrieval number and where **citation-weighted re-ranking**
  (`apply_citation_boost`, `cross_encoder_rerank`) shows up: helpful re-ranking
  raises nDCG, burying a highly-relevant paper lowers it.
  - *Soundbite:* MRR asks how soon the first good result appears; nDCG asks how
    good the whole ordering is.

**Classic IR (recall@k / precision@k) vs. RAGAS "context recall / precision".**
These terms overlap loosely but differ mechanically:

| | recall@k / precision@k | context recall / precision (RAGAS strict) |
|---|---|---|
| Unit | whole documents / PMIDs | claims / sentences of the answer |
| "Relevant" judged by | pre-labeled gold doc set (ID match) | dynamically, usually by LLM, against the ground-truth answer |
| Needs an LLM | no — deterministic | yes |
| Ranking-sensitive | precision@k ignores order within top-k | context precision rewards relevant chunks ranked higher |
| Tier | 1 (blocking) | 2 (judge) |

The key difference: recall@k asks *did the document IDs match*; context recall
asks *is the information needed to answer present at all* — so context recall can
be 1.0 even when an ID was missed, if that paper's information is also carried by
another retrieved passage. **Decision: Tier 1 uses classic recall@k / precision@k
/ nDCG (deterministic, cheap). We do not add RAGAS context recall separately —
its intent overlaps groundedness + completeness, which we measure with clearer
semantics; avoiding metric redundancy.**

### B. Grounding & attribution (Answer ↔ Context) — the safety core

- **Groundedness / Faithfulness** — of the claims the answer makes, the fraction
  supported by *some* retrieved passage. Decompose the answer into atomic claims;
  a judge checks each against context. Treats context as one bag — it does **not**
  check which citation was attached. Catches **hallucination**.
- **Attribution precision** — of the (claim → cited PMID) pairs, the fraction
  where that PMID genuinely supports the claim. Catches **misattribution**.
- **Attribution recall** — of claims that *require* a citation, the fraction that
  have one. Catches **uncited assertions**.

**Groundedness vs. attribution — why both.** Groundedness asks *is it in the
evidence*; attribution asks *is it in the evidence I pointed you to*. They can
diverge — the dangerous case is **grounded but mis-attributed**: the fact is
genuinely in a retrieved passage (groundedness ✓), but the model cited the wrong
paper for it (attribution ✗). A physician who clicks the citation to verify hits
a paper that doesn't say it. Groundedness alone cannot catch this; attribution is
a separate, stricter gate.

| Case | Model did | Groundedness | Attribution |
|---|---|---|---|
| Cited the right paper | correct link | ✓ | ✓ |
| Cited a real but wrong paper | info is in context, citation isn't | ✓ | ✗ (misattribution) |
| Made the claim, no citation | info in context | ✓ | ✗ (recall) |
| Fabricated + slapped a citation | not in context | ✗ | ✗ (precision) |

### C. Answer quality (Answer ↔ Query)

- **Answer relevance** — does the answer actually address the question (not
  wander, not over-answer). The missing triad leg: grounded ≠ responsive.
- **Completeness / coverage** — of the key evidence points in the reference
  answer, how many the response covered. Catches **one-paper answers**.

### D. Safety / robustness

- **Abstention / negative rejection** — on questions where the corpus lacks good
  evidence, does the system decline / hedge instead of fabricating. Measured on
  the adversarial gold cases; partly deterministic (did it emit citations it
  shouldn't have?) plus judge for tone. The core physician-safety behavior.
- **Noise robustness** *(defer to v2)* — inject an irrelevant passage; does
  faithfulness hold?
- **Contradiction surfacing** *(defer to v2)* — when retrieved passages conflict,
  does the answer present the disagreement rather than cherry-pick?

**Mental model:** retrieval metrics ask *did we find and rank the right
evidence*; groundedness asks *did we stick to it*; attribution asks *did we point
to the right piece of it*; answer relevance asks *did we answer the question*;
abstention asks *did we know when to stay silent*.

---

## 6. Why the two tiers are separate

The deciding question for any metric is: **is it fast, cheap, and
deterministic?** Only then can it be a merge gate. Tier 2 fails all three, so it
can only be trend monitoring, not a blocking gate.

**Why Tier 2 (LLM judge) cannot gate PRs:**
- **Slow** — generate the full answer, then judge each claim across dimensions;
  minutes per run. A PR gate must return in seconds/low-minutes.
- **Costly** — every run burns tokens (generation + judging). Per-push on every
  PR is unaffordable; nightly is once/day and predictable.
- **Non-deterministic** — the judge scores 0.91 today, 0.89 tomorrow on identical
  code. Gating merges on a noisy score produces flaky red builds; developers
  learn to blind-retry or ignore CI, destroying its credibility. *Putting a noisy
  metric on the critical path breaks the whole gate.*

**What nightly buys:**
- Catches regressions **not in any single PR diff** — index rebuilds, corpus
  updates, model/provider version drift, and the cumulative effect of many
  individually-harmless prompt tweaks. No single PR "owns" these, so a PR gate
  never sees them.
- Produces a **trend line** you can bisect against dates, not a red/green verdict.

**Full triggering strategy:**

| | Trigger | Blocks merge | Output |
|---|---|---|---|
| Tier 1 | every PR | **yes** | red/green |
| Tier 2 | nightly cron + `eval` label on high-risk PRs | no | trend line + alerts + PR comment |

*Analogy:* Tier 1 is unit tests / compile — fast, deterministic, must be green to
merge. Tier 2 is a performance benchmark / canary — slow and noisy; you watch the
trend and alert thresholds, never gate a single run on it.

---

## 7. Ground-Truth Data

Labels are **LLM-bootstrapped + human spot-checked**, then **frozen as committed
JSONL**. CI runs against the static, versioned set so scores are comparable
across commits.

```
evals/gold/
  questions.jsonl     # one record per gold question (schema below)
  aliases.jsonl       # normalization gold pairs
# therapy gold already exists at data/seeds/therapy_gold.json
```

**`questions.jsonl` record:**
```json
{
  "id": "q-tofersen-sod1",
  "question": "What's the evidence for tofersen targeting SOD1?",
  "category": "single-target",
  "expected_entities": ["SOD1", "tofersen", "oxidative-stress"],
  "relevant_pmids": ["12345678", "23456789"],
  "reference_answers": ["...", "..."],
  "reference_citations": ["12345678", "23456789"],
  "verified": true,
  "verified_by": "sme|null",
  "notes": "adversarial: weak evidence expected"
}
```

**Coverage target: 20–50 questions** spanning:
- single-target, mechanism, comparative, and trial-status questions;
- **adversarial / negative cases** — questions with weak or no corpus evidence
  (does the system hedge instead of fabricate?) and out-of-scope questions.

**Label-quality guardrails (because labels are model-drafted):**
- **Bootstrap labels with a different strong model than the one that answers
  queries** — otherwise we grade the student with its own answer key.
- **`verified` flag per record.** Trust verified labels; treat unverified as
  weak signal. Track spot-check coverage as a headline metric.
- **Judge calibration (one-time, before trusting Tier 2):** SME grades ~20
  answers; measure judge–human agreement per dimension; trust the judge only on
  dimensions where it agrees.
- **Freeze on commit.** Never regenerate labels inside a CI run.

---

## 8. Harness Structure

```
evals/
  gold/
    questions.jsonl
    aliases.jsonl
  runner.py           # orchestrates layers; emits per-layer + aggregate scores
  layers/
    citations.py      # Tier 1 — deterministic citation validity gate
    retrieval.py      # Tier 1 — recall@k / MRR / nDCG
    kg_expansion.py   # Tier 1 — expansion overlap
    normalization.py  # Tier 1 — alias collapse
    landscape.py      # Tier 1 — classifier accuracy vs therapy_gold.json
    synthesis.py      # Tier 2 — LLM-judge faithfulness / relevance
    end_to_end.py     # Tier 2 — holistic
  judge.py            # pinned judge model wrapper
  report/             # scored JSON + trend history (committed or dashboarded)
```

**Reporting rules.**
- **Per-layer scores, never a single aggregate number** — a drop must point to a
  stage.
- **Record the corpus/graph hash** in every report, so a re-index effect is not
  misattributed to a prompt change.
- **Thresholds per metric** (see §9); Tier 1 fails the build below threshold.
- **Trend history** for Tier 2 so gradual drift is visible.

---

## 9. Metrics and Thresholds (initial — tune after baseline)

| Layer | Metric | Initial threshold | Tier |
|---|---|---|---|
| Citations | % valid cited PMIDs | 100% (hard) | 1 |
| Retrieval | recall@10 | ≥ 0.80 | 1 |
| Retrieval | MRR | ≥ 0.60 | 1 |
| KG expansion | expected-entity recall | ≥ 0.85 | 1 |
| Normalization | alias-collapse accuracy | ≥ 0.95 | 1 |
| Landscape | macro-F1 | ≥ 0.75 | 1 |
| Synthesis | faithfulness (judge) | ≥ 0.90, alert on drop | 2 |
| Synthesis | relevance / completeness | trended | 2 |
| End-to-end | holistic score | trended | 2 |

Thresholds are placeholders. **Set them from the first baseline run**, at or
just below observed performance, so CI catches regressions without blocking on
day one.

---

## 10. Build Order

1. **Citation validity gate + therapy classifier.** Deterministic,
   self-contained, highest value-per-effort. Ship as Tier 1 immediately.
2. **Retrieval + KG-expansion + normalization metrics.** Requires drafting the
   gold `questions.jsonl` / `aliases.jsonl` first.
3. **Tier-2 judge** (synthesis + end-to-end). Last — after judge calibration.

---

## 11. Open Questions

- **Judge model choice + version pin** for Tier 2 — which model, and how do we
  version it so scores stay comparable?
- **Where does trend history live** — committed JSON in `evals/report/`, or a
  lightweight dashboard?
- **CI cadence for Tier 2** — nightly cron, or an `eval` PR label, or both?
- **SME availability** for the one-time judge calibration and periodic spot-checks.
- **Gold set size** — start at 20 and grow, or invest in 50 up front?
