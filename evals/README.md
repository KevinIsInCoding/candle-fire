# evals — Tier-1 red/green eval gate

Deterministic pass/fail checks that guard the query pipeline against regressions.
Design and rationale: [`docs/eval-plan.md`](../docs/eval-plan.md).

## Two suites

| Suite | Runs where | Needs | Checks |
|---|---|---|---|
| `offline` | PR CI, every push (`.github/workflows/eval-gate.yml`) | stdlib only — no data, no LLM, no network | gold schema, normalization alias-collapse, therapy-landscape gold |
| `data` | deploy gate, before `git push hf main` | local ChromaDB index + graph pickle | citation existence, KG-expansion recall, retrieval recall@k / precision@k / MRR |
| `judge` (Tier 2) | nightly / on-demand (`.github/workflows/eval-tier2.yml`) | runtime data **+ `ANTHROPIC_API_KEY`** | groundedness, attribution precision/recall, answer relevance, abstention*, completeness* |

`offline` + `data` are the deterministic **red/green gate** (`--suite all`). The
`judge` suite is **Tier 2**: it runs the real agent + a pinned judge model (Opus 4.8,
different from the Sonnet synthesizer), so it is non-deterministic and costs tokens.
It runs nightly, alerts on threshold drops, and **never blocks a PR** — it is not part
of `all`. *(abstention/completeness are reported only when the gold set has adversarial
questions / reference answers.)*

The offline/data split exists because the retrieval/graph assets are gitignored
(uploaded to a HF dataset repo, not committed), so retrieval-side checks can only run
where the data lives — at deploy — while the data-free checks gate every PR.

## Run

```bash
uv run python -m evals.runner --suite offline   # PR CI
uv run python -m evals.runner --suite data      # deploy gate
uv run python -m evals.runner --suite all       # offline + data (the red/green gate)
uv run python -m evals.runner --suite judge      # Tier-2 LLM judge (needs data + ANTHROPIC_API_KEY)
```

Exit 0 = green, exit 1 = red (any check failed). A `SKIP` never fails the build —
it means a check had nothing to run (e.g. a gold question with no labeled PMIDs
yet). A JSON report is written to `evals/report/latest.json` (gitignored) for
trend tracking.

## Gold data (`gold/`)

- `aliases.jsonl` — entity alias groups; every surface form must collapse to one `canonical_id`.
- `questions.jsonl` — physician questions with `query_entities`, `expected_entities`
  (KG-expansion recall), and binary `relevant_pmids` (retrieval). Labels are
  LLM-bootstrapped + spot-checked; `verified: true` once an SME confirms.

**Growing the set (option C):** add questions with `expected_entities` first (KG
check runs immediately); add `relevant_pmids` when labeled (retrieval check
activates per-question — unlabeled questions skip, never fail). Graded relevance /
nDCG is deferred until the set is larger.

## Thresholds

All in [`thresholds.py`](thresholds.py), set at/just below the current baseline so
the gate catches regressions without blocking on day one. Re-baseline deliberately
(a reviewable line in a PR) when the pipeline genuinely improves.
