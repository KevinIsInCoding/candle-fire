# Migrating Candle-Fire to AWS

**Document type:** 6-page narrative · **Author:** _<you>_ · **Reviewers:** _<add>_
**Status:** Draft · **Date:** 2026-09-16
**Service:** candle-fire — physician-facing ALS research intelligence tool

---

## 1. Summary

Candle-fire runs today as a Gradio application on a HuggingFace Space. It has
served us well as a prototype, but it has two limitations we can no longer live
with: the HuggingFace serving layer is fragile (it caused a full outage in July
2026, documented in our COE), and it gives us no path to two things we may want
later — a HIPAA-compliant LLM path for a physician-facing tool, and a
self-hosted model we can learn to serve.

This document proposes migrating candle-fire to AWS as **three independent,
sequential stages**, ordered by effort and value:

- **Stage 1 — Lift-and-shift to a single EC2 box (do this now).** Move the app
  as-is onto one cheap, always-on CPU instance with data on persistent storage
  and a custom HTTPS domain. This alone eliminates both causes of the July
  outage. The LLM path is unchanged (direct Anthropic API). ~$40–55/month,
  ~½ day of work, no application code changes.
- **Stage 2 — Switch the runtime LLM to Amazon Bedrock (deferred).** Bring the
  physician-query path under the AWS BAA for HIPAA compliance. Small code
  change; adds a few dollars/month.
- **Stage 3 — Self-host an open model on a scale-to-zero GPU (deferred).** Learn
  LLM serving and begin replacing the managed API on the lowest-risk pipeline
  steps, without paying for an idle GPU.

**We are executing Stage 1 first.** Stages 2 and 3 are documented here so the
sequence is legible and Stage 1 does not foreclose them, but they are explicitly
out of scope until Stage 1 is live and stable. The rest of this document details
Stage 1 and sketches 2 and 3.

---

## 2. Goals and Tenets

**Goals.**

1. The website is available 24 hours a day, on a custom domain, over HTTPS. *(Stage 1)*
2. We eliminate the serving-layer fragility that caused the July 2026 outage. *(Stage 1)*
3. Cost stays low and scales with actual usage, not with idle capacity. *(all stages)*
4. The LLM path used for physician queries can be made HIPAA-compliant. *(Stage 2)*
5. We can self-host and serve an open LLM. *(Stage 3)*

**Tenets (what we optimize for when goals conflict).**

- **Ship the smallest valuable thing first.** Getting off HuggingFace onto
  reliable, cheap infra is worth doing on its own, before any LLM changes.
- **Only the website must be up 24/7 — the LLM does not.** This distinction is
  what makes the eventual GPU stage cheap; it is the backbone of Stage 3.
- **Frugality over convenience.** At ~10 conversations/day, a GPU running 24/7
  would be idle ~99% of the time. We would rather accept a cold-start delay
  than pay for idle silicon — which is why self-hosting is deferred, not rushed.
- **Reversible steps.** Each stage is independently valuable and can be paused
  or rolled back without undoing the previous one.

---

## 3. Background: What Candle-Fire Is Today

Candle-fire answers free-text ALS research questions by combining a NetworkX
knowledge graph with retrieval over ~10,000 curated paper abstracts in
ChromaDB, then synthesizing a cited answer with Claude. The facts that matter
for this migration:

- **Compute profile is light and mostly CPU-bound.** The web app, ChromaDB
  (~1.2 GB), the knowledge graph, the BioLORD embedding model, and the
  cross-encoder reranker all run comfortably on CPU. Total data footprint is
  ~1.4 GB.
- **The LLM is the only external dependency at query time.** Synthesis uses
  `claude-sonnet-4-6` via the direct Anthropic API today. Offline pipeline
  steps use Haiku (extraction) and Opus (landscape classification).
- **Traffic is very low and bursty.** Roughly 10 conversations per day —
  under one request every two hours on average, with idle stretches.
- **The current host is the weak point.** The July 2026 outage had two causes:
  HuggingFace's experimental server-side rendering layer, and the fact that
  every cold rebuild re-downloaded the ~1.3 GB index (`app._ensure_data()`
  pulls the corpus from a HF dataset repo when local storage is empty). The
  application itself was healthy throughout.

The migration is therefore not about scaling for load — it is about control,
reliability, and keeping the bill small, while leaving the door open to
compliance and self-hosting later.

---

## 4. Stage 1 — Lift-and-shift to EC2 (the work we are doing now)

### 4.1 What Stage 1 is (and is not)

**Is:** move the app unchanged onto one always-on CPU EC2 instance; put `data/`
on a persistent EBS volume so it is fetched once and never re-downloaded;
terminate HTTPS on a custom domain. This fixes both outage causes.

**Is not:** any LLM change. The synthesis path stays on the direct Anthropic
API. No Bedrock, no GPU, no self-hosting. Application code is unchanged —
`app._ensure_data()` already no-ops when the data is present on disk, so on a
persistent volume it downloads the 1.4 GB corpus on first boot and never again.

### 4.2 Target architecture

```
              :443 HTTPS                      :7860 (localhost only)
Users ──► Caddy (auto Let's Encrypt)  ──►  Gradio (ssr_mode=False)
              on the EC2 box                Chroma + KG + BioLORD + cross-encoder
                                           data/ on persistent EBS
                                                   │
                                                   └──► Anthropic API (unchanged)
```

One `t4g.large` box runs everything. Caddy terminates TLS and reverse-proxies
to Gradio on `127.0.0.1:7860`; port 7860 is never exposed publicly.

### 4.3 Decisions

| Decision | Choice | Why |
|---|---|---|
| **Instance** | `t4g.large` (2 vCPU / 8 GB, Graviton/ARM) | Cheapest that fits Chroma 1.2 GB + torch + BioLORD + reranker in RAM. All deps have arm64 wheels. Fallback: `t3.large` (x86) if an arm64 wheel bites. |
| **Disk** | 30 GB `gp3` root EBS | OS + `.venv` + torch/model caches (~5 GB) + 1.4 GB data, with headroom. EBS persists across stop/start → no re-download. |
| **Data load** | Reuse `_ensure_data()` HF download on first boot | Zero code change. Downloads 1.4 GB once to EBS; no-ops forever after. `HF_TOKEN` needed only if the data repo is private. |
| **TLS/domain** | Caddy + a DNS A record | Auto-provisions & renews Let's Encrypt certs; ~$1–2/mo vs ~$18/mo for an ALB we do not need at 10 conv/day. |
| **Process mgmt** | `systemd` unit running `uv run python app.py` | Auto-restart on crash, auto-start on boot. |
| **Secrets** | `.env` on box, `chmod 600`, owned by the service user | Lowest effort. Stage 2 replaces the key with an IAM role. |
| **Network** | SG: 443 + 80 world, 22 your-IP-only, 7860 closed; Elastic IP | Caddy needs 80 for the ACM HTTP-01 challenge + 80→443 redirect. Elastic IP is a stable DNS target across restarts. |

### 4.4 Runbook (~½ day)

Scripted in [`../deploy/`](../deploy/) (`bootstrap.sh`, `candle-fire.service`,
`Caddyfile`). See [`../deploy/README.md`](../deploy/README.md).

1. **Provision** `t4g.large`, Ubuntu 24.04, 30 GB gp3, key pair, the security
   group above. Allocate + associate an **Elastic IP**.
2. **DNS:** A record `candle-fire.<domain>` → Elastic IP.
3. **Bootstrap:** `sudo DOMAIN=candle-fire.<domain> bash deploy/bootstrap.sh`
   (installs uv, Caddy, service user, deps, systemd unit, Caddy config).
4. **Secrets:** set `ANTHROPIC_API_KEY` in `/opt/candle-fire/.env`.
5. **Start + first data pull:** `systemctl start candle-fire` → `_ensure_data()`
   pulls 1.4 GB to EBS once; watch `journalctl -u candle-fire -f`.
6. **Cut over:** verify HTTPS + a real query end-to-end. **Keep the HF Space
   alive as fallback** for a few days before decommissioning — instant rollback.

### 4.5 Cost (Stage 1, us-east-1, on-demand)

| Component | Monthly |
|---|---|
| `t4g.large` 24/7 | ~$49 on-demand → **~$30** with a 1-yr Savings Plan |
| 30 GB gp3 EBS | ~$2.40 |
| Elastic IP (while attached) | $0 |
| Caddy TLS + DNS hosted zone | ~$1 |
| Anthropic API (unchanged, ~10 conv/day) | ~$5–15 |
| **Total** | **~$40–55/mo on-demand, ~$25–35 with a Savings Plan** |

### 4.6 Risks (Stage 1)

| Risk | Mitigation |
|---|---|
| 8 GB RAM too tight (torch + BioLORD + Chroma + reranker) | Load-test in step 5 with `htop`; bump to `t4g.xlarge` (16 GB) — a one-line instance-type change. |
| arm64 wheel gap for a dependency | All current deps ship arm64 wheels; verify during `uv sync`. If one bites, switch to `t3.large` (x86). |
| First-boot HF download fails (private repo / rate limit) | `HF_TOKEN` in `.env`; or `scp`/`aws s3 cp` the `data/` dir directly. |
| Single point of failure | Acceptable at this traffic; weekly EBS snapshot makes a rebuild a ~15-min restore. |

---

## 5. Stage 2 — Anthropic API → Amazon Bedrock (HIPAA) *(deferred)*

For a physician-facing tool, queries may contain PHI. Bedrock is HIPAA-eligible
under the **AWS BAA**, so routing Claude calls through Bedrock brings the managed
LLM path into compliance. The code change is small because `llm.py` already
abstracts the provider behind an `LLMProvider` interface: we add a
`BedrockProvider` (same `.messages.create` surface) and select it via the
existing `LLM_PROVIDER` environment variable.

Key implementation facts, for when we pick this up:

- **Model IDs carry an `anthropic.` prefix** on Bedrock — e.g.
  `anthropic.claude-sonnet-4-6`. A bare `claude-*` ID returns a 400.
- **Cross-region inference profiles** (e.g. `us.anthropic.claude-sonnet-4-6`)
  route across US regions with automatic failover; confirm exact profile IDs in
  the Bedrock console.
- **Auth is IAM, not an API key.** Attach an IAM role to the instance granting
  `bedrock:InvokeModel` (+ streaming) on the Claude model ARNs — replacing the
  `.env` key from Stage 1.
- **Bedrock has no Batches API.** `scripts/build_landscape.py` (offline, no PHI)
  stays on the direct Anthropic API. Only the runtime Q&A path moves.
- **Bedrock supports *manual* `cache_control`** (which `llm.py` already uses via
  `cached_system`), just not *automatic* caching. No change required.

HIPAA is a property of the whole stack, not just the LLM call: it also requires
KMS encryption at rest (EBS/S3), TLS in transit (Stage 1's domain), access
logging, and PHI-free application logs — tracked as explicit line items when we
do this stage. Estimated effort ~1–2 days; added cost a few $/month.

---

## 6. Stage 3 — Self-hosted model on a scale-to-zero GPU *(deferred)*

The eventual self-hosting goal serves a 7–8B open model (Qwen2.5-7B-Instruct or
Llama-3.1-8B-Instruct, AWQ-quantized for a 16 GB T4) via **vLLM**
(OpenAI-compatible endpoint) on a GPU that **scales to zero** when idle
(SageMaker Async Inference at min-capacity 0, or an on-demand `g4dn.xlarge`
started per session). A `SelfHostedProvider` in `llm.py` reuses the provider
pattern. The accepted trade-off is a **1–3 min cold start** on the first
question after idle, covered by falling back to the managed API during warm-up.

Rollout when we get here: (a) stand it up as a hidden sandbox; (b) route the
lowest-risk step — query **entity-extraction** — to it while synthesis stays on
the managed API, shadow-logging both for a week; (c) promote further (canary
"easy" questions with automatic fallback) only as trust is earned. This is why
the tenet "only the website must be up 24/7" matters: the GPU is off ~99% of the
day, so a 24/7 GPU (~$250–384/mo) is the frugality trap we avoid.

---

## 7. Options Considered (why this ordering)

We evaluated four end-states against effort, performance, cost, and
self-hosting. The three stages above are a *path through* this table, not a
single pick: **Stage 1 = Path A**, **Stage 2 = Path A + Bedrock**, **Stage 3
reaches Path B.**

| Path | Description | Effort | ~Cost/mo | Self-host? |
|---|---|---|---|---|
| **A. Lift-and-shift, managed LLM** ← *Stage 1* | One CPU box runs everything; LLM stays a managed API | Low | $40–55 | No |
| **B. Split: 24/7 CPU web + scale-to-zero GPU** ← *Stage 3 target* | Web on cheap CPU; self-hosted model on a GPU that scales to zero | Medium | $45–90 | **Yes** |
| C. Self-host small model on CPU | One larger CPU box; small quantized model, no GPU | Medium | $80–100 | Yes (modest) |
| D. All-in GPU 24/7 | Single GPU box runs everything, always warm | Low–Med | $250–384 | Yes |

Path A is the cheapest and fastest and delivers the outage fix on its own —
which is exactly why it is Stage 1. Path C (CPU-only model) is both slow and
lower quality, and Path D pays continuously for an idle GPU; both are rejected.
Path B is the eventual target, reached incrementally via Stages 2 and 3 rather
than as a big-bang cutover.

---

## 8. FAQ

**Why lift-and-shift first instead of building the full split architecture?**
Getting off the fragile HuggingFace serving layer onto reliable, cheap infra is
independently valuable and low-risk. It fixes the outage class today, changes no
application code, and does not foreclose Bedrock or self-hosting. Doing it first
means we stop the bleeding before taking on the larger compliance and GPU work.

**Does Stage 1 lock us out of HIPAA or self-hosting?**
No. `llm.py`'s provider abstraction means Stage 2 is an additive
`BedrockProvider`, and Stage 3 an additive `SelfHostedProvider`. Stage 1's box,
domain, and EBS are all reused.

**Why not run the GPU 24/7 so there's no cold start?**
At 10 conversations/day that is ~$250–384/month to keep a GPU warm for minutes
of daily use. The cold start is the deliberate, frugal trade-off — and it only
becomes relevant in Stage 3, which is deferred.

**Does candle-fire actually handle PHI?**
The corpus (papers, graph) is not PHI. The risk is in free-text physician
queries, which could contain patient context. Stage 2 treats the query path as
PHI-bearing; until then, avoid entering PHI.

**How much application code changes in Stage 1?**
None. It is pure infrastructure: EC2 + EBS + systemd + Caddy + DNS.

---

## 9. Appendix: Next Actions

**Stage 1 (now):**
1. Confirm the inputs in §10 (domain, AWS region, repo access, HF data-repo visibility).
2. Provision `t4g.large` + 30 GB gp3 + Elastic IP + security group.
3. Point DNS at the Elastic IP.
4. Run [`deploy/bootstrap.sh`](../deploy/bootstrap.sh); set `ANTHROPIC_API_KEY`; start the service.
5. Verify HTTPS + a real query; keep the HF Space as fallback for a few days.

**Later:** Stage 2 (`BedrockProvider` + IAM role + AWS BAA), then Stage 3
(vLLM scale-to-zero + `SelfHostedProvider`).

---

## 10. Inputs Needed to Execute Stage 1

Before provisioning, we need:

1. **Domain** — the hostname to serve on (e.g. `candle-fire.example.com`) and
   where DNS is managed (Route 53 or an external registrar).
2. **AWS region** — recommended `us-east-1` (matches cost table; Bedrock model
   availability in Stage 2).
3. **Repo access** — the box clones `github.com/KevinIsInCoding/candle-fire`.
   Confirm it is public, or provide a deploy key / PAT if private.
4. **HF data-repo visibility** — is `KevinIsCoding/candle-fire-data` public or
   private? If private, an `HF_TOKEN` with read access.
5. **`ANTHROPIC_API_KEY`** — the key the box will use (can be the existing one).
6. **SSH access** — an EC2 key pair, and your public IP for the port-22 SG rule.
7. **Instance arch decision** — default `t4g.large` (Graviton, cheapest) unless
   you prefer `t3.large` (x86) to avoid any arm64 surprises.
