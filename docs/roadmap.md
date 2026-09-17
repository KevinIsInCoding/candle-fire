# Candle-Fire — Roadmap / To-Do

Future improvements, with enough context to pick up (or explain) later.

---

## 1. KG-augmented query rewriting for semantic search

**Status:** idea / not started · **Type:** retrieval quality · **Effort:** small–medium

### The problem
Semantic (vector) search embeds the physician's **raw question** into one vector.
When the question is short, vague, or uses sparse phrasing ("What's the evidence
for tofersen?"), that vector can miss mechanistically-relevant papers — e.g. the
SOD1-biology and neurofilament-biomarker papers that are the real evidence but
never mention the drug name.

### Current design (and why it's built this way)
The knowledge graph **already** feeds retrieval — but as a *separate* signal, not
by touching the semantic query:
- `semantic search` → embeds the raw question (kept faithful to user intent).
- `entity search` → embeds each KG-**expanded** entity separately.
- `keyword search` → exact `$contains` match for drug codes / gene IDs.
- All three are fused with **RRF**.

Keeping them separate is deliberate: if KG expansion is noisy, a bad entity only
adds one weak entity-query that RRF down-weights — it never corrupts the one query
that stays faithful to what the user actually asked.

### The proposed improvement
Also use the KG to **rewrite/enrich the semantic query itself**: append a few
high-confidence canonical terms (target, mechanism, biomarker) so the semantic
vector encodes KG knowledge too, not just the user's phrasing.

Example: `"What's the evidence for tofersen?"`
→ `"tofersen SOD1 antisense oligonucleotide neurofilament ALS evidence"` → embed.

### Risks to manage
- **Dilution** — too many terms average the vector into mush. Must be selective.
- **Topic drift** — noisy expansion drags retrieval off-topic. (Today `tofersen,
  SOD1` expands to **512** entities incl. "methylmercury", "CAR-T" — far too broad.)

### Prerequisite
Tighten KG expansion first: rank neighbors by **edge confidence**
(`KG_MIN_EDGE_CONFIDENCE`), cap to **top-k (~3–5)**, prefer **1-hop** direct
relations (target / mechanism / biomarker). This also improves entity search on
its own.

### Plan (measure, don't guess)
1. Tighten KG expansion to top-k high-confidence neighbors.
2. Add a **feature flag** for "KG-rewritten semantic query" (default off).
3. Run the **Tier-1 / Tier-2 eval suite** (retrieval recall + answer quality) with
   the flag on vs off.
4. Ship the default only if the numbers improve; otherwise roll back.

### Why it's a good story
It's a real IR technique (query expansion / generation-augmented retrieval), it
forces an explicit **architecture trade-off** (fuse KG as a separate RRF signal vs.
rewrite the query), and the decision is made by **evaluation**, not intuition —
which is exactly what the eval harness was built for.

---

## 2. Chat latency — instance upsize (parked until funding)

**Status:** deferred · **Type:** performance / cost

First-token latency is ~19s, broken down as: ~1.3s tool call + ~9s **CPU
retrieval** (dominated by ~5.3s cross-encoder rerank on 2 vCPUs) + ~8s Bedrock
synthesis TTFT. The retrieval half is CPU-bound and provider-independent.

Lever: resize `t4g.large` → `t4g.xlarge` (4 vCPU) — pure ~2× speedup on all CPU
work (rerank, embeddings, trials warm-up), no quality loss, ~+$49/mo on-demand
(~+$12–15/mo on a 1-yr Savings Plan). One-line resize + reboot (snapshots exist).
Free alternative: `RRF_TOP_N` 20→12 (~2s, small recall cost). Parked until funding.
