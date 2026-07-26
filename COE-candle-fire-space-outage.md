# Correction of Errors (COE): candle-fire HuggingFace Space Outage

> **Status:** Draft · **Owner:** _<you>_ · **Reviewers:** _<add>_
> **Service:** candle-fire — physician-facing ALS research intelligence tool (Gradio app on a HuggingFace Space)
> **Severity:** _<assign>_ · **Date of incident:** 2026-07-23/24 · **Author:** _<you>_

## 1. Summary

While deploying a corrected (de-duplicated) knowledge-graph dataset to the
candle-fire HuggingFace Space, the Space became intermittently and then fully
unavailable, returning HTTP 403 / 503 / 500 to users. The underlying Gradio
application was healthy the entire time (it booted and loaded data on every
attempt); the failure was in the **serving layer** — Gradio's experimental
server-side rendering (SSR) layer, which is unstable on this Space and was
exposed when a forced restart triggered a cold rebuild. The incident was
prolonged by the remediation process itself (repeated rebuilds/restarts while
diagnosing). Service was restored by disabling Gradio SSR (`ssr_mode=False`) and
pinning the Gradio version.

## 2. Customer / User Impact

- The candle-fire Space (a physician-facing research tool) returned errors
  (403/503/500) for the duration of the incident — from intermittent failures
  to a period of complete unavailability (all requests 503).
- No data was lost or corrupted. The corrected dataset deployed successfully;
  the outage was purely in application serving.
- Duration: approximately _<fill in>_ (spanned the diagnosis + several rebuild
  cycles). Each cold rebuild re-downloaded a ~1.3 GB index, adding multi-minute
  unavailability windows.

## 3. Timeline (ordered; times approximate)

| Phase | Event |
|---|---|
| Deploy start | Corrected graph pickle uploaded to the `candle-fire-data` dataset repo (the Space downloads data from here at startup). |
| Trigger | `factory_reboot=True` issued on the Space to force it to re-download the corrected pickle (its `_ensure_data()` only downloads "if not present"). This wiped the Space's storage and forced a full cold rebuild. |
| Data-to-git | Deduped data files pushed to the Space git repo; initial pushes rejected by HuggingFace's 10 MB non-LFS file limit (files had grown to ~19–20 MB); resolved by moving them to Git LFS. |
| First symptoms | Space began returning 403 → 503 → 500 intermittently (~2/10 success). |
| Diagnosis 1 | Identified Gradio's experimental Node SSR as the failing layer. Deployed `ssr_mode=False`; success rate improved (~partial), but measurements were taken during rebuild churn and were noisy. |
| Regression | A second `factory_reboot` (remediation attempt) moved the Space from partial to **fully down** (0/12, all 503). |
| Wrong turn | Reverted `ssr_mode=False` on a "restore known-good config" assumption; this re-enabled SSR and kept the Space fully down (0/12). |
| Correction | Log evidence showed SSR-on → 503 vs SSR-off → recovers. Re-applied `ssr_mode=False` and pinned `gradio==6.14.0`; **stopped rebuilding** and allowed the Space to fully settle. |
| Resolved | After the post-boot warmup window, the Space held **12/12 HTTP 200**. Fix backfilled to git and synced to `main`. |

## 4. Root Cause

**The application depended on Gradio's experimental SSR serving layer (enabled by
default in Gradio 6.x), which is unstable on this Space. This dependency was
unvalidated and had been masked because the Space had been serving from warm/
persistent state and had not cold-started under that configuration — until a
forced `factory_reboot` (required to pick up new data) triggered a clean cold
rebuild that exposed it.**

### 5 Whys
1. **Why was the Space down?** HuggingFace's proxy returned 503 — the app was
   running on port 7860 but was unreachable through the serving layer.
2. **Why was it unreachable?** Gradio's experimental Node SSR layer failed to
   serve on this Space.
3. **Why did SSR fail now and not before?** A forced factory-reboot cold-started
   the app under the default (SSR-on) configuration; the Space had previously
   been running on warm state and had never cleanly cold-started under it.
4. **Why was a factory-reboot needed at all?** The data-refresh path
   (`_ensure_data()` downloads data only if it is not already present on disk)
   required wiping Space storage to force a re-download of the corrected pickle.
5. **Why did the app rely on an unstable serving mode?** Gradio 6.x turns on
   experimental SSR by default; it was never explicitly disabled, pinned, or
   validated for this Space.

## 5. Detection

- Detection was **user-reported** (errors observed directly on the Space URL).
- There was no automated health check / uptime probe on the Space that would
  have caught the regression at deploy time.

## 6. Resolution

- Set `ssr_mode=False` in `demo.launch()` (bypass Gradio's experimental SSR;
  serve the classic client-rendered app directly).
- Pinned `gradio==6.14.0` in `requirements.txt` and `pyproject.toml` to match the
  Space's README `sdk_version` and prevent version drift on future rebuilds.
- Allowed the Space to fully settle after the final deploy before validating,
  and confirmed 12/12 HTTP 200.
- Backfilled the fix to `origin/hf-clean`, synced it to `main` (PR #15), and
  realigned `origin/hf-clean` with the Space's `hf/main` history so future
  deploys are clean fast-forwards.

## 7. What Went Wrong in the Response (contributing factors)

- **Over-churning:** Multiple `factory_reboot`s and rebuilds during diagnosis
  each forced a ~1.3 GB cold-start re-download, extending downtime and adding
  noise. One `factory_reboot` moved the Space from partial to fully down.
- **Measuring during warmup:** Success-rate probes were run during rebuild/
  warmup windows, producing misleading partial results and one false "worse".
- **Reverting a working mitigation:** `ssr_mode=False` was reverted on an
  assumption ("restore known-good") rather than evidence, which re-broke it.
- **No staging/validation before prod:** Changes were validated directly on the
  production Space.

## 8. Action Items

| # | Action | Type | Owner | Status |
|---|---|---|---|---|
| 1 | Disable Gradio SSR (`ssr_mode=False`) and pin `gradio==6.14.0` | Fix | _<you>_ | ✅ Done |
| 2 | Redesign data refresh so updating the dataset repo does **not** require a `factory_reboot` / full cold rebuild (e.g., version the data path or use a lighter refresh signal) — removes the destabilizing trigger and the 1.3 GB re-download | Prevent | | Todo |
| 3 | Pin all runtime-critical dependencies (no `>=,<` ranges that drift on rebuild); keep `requirements.txt` in sync with README `sdk_version` | Prevent | | Todo |
| 4 | Write a deploy runbook: two deploy targets (dataset repo for data, Space git for code), the LFS >10 MB requirement, the cold-start warmup window, and how to verify (probe for stable 200s) | Prevent | | Todo |
| 5 | Add post-deploy verification that waits out the warmup window before declaring success (don't measure during rebuild) | Detect | | Todo |
| 6 | Add an automated uptime/health probe on the Space with alerting | Detect | | Todo |
| 7 | Incident-response guidance: during diagnosis, change one variable at a time, avoid repeated factory-reboots, and don't revert a working mitigation without evidence | Process | | Todo |
| 8 | Reduce cold-start fragility (the ~1.3 GB index download) — evaluate persistent storage strategy or a smaller/streamed index | Prevent | | Todo |
| 9 | Consider a staging Space to validate rebuilds before touching production | Prevent | | Todo |

## 9. Corrective Actions — Concrete Changes (Action Items 2–4)

### Item 2 — Data refresh without a factory-reboot

**Problem.** `_ensure_data()` short-circuits when the files already exist
(`if not need_chroma and not need_graph: return`), so a running Space never
picks up updated data. That is what forced the `factory_reboot` (storage wipe),
which cold-rebuilt the image and exposed the SSR failure.

**Code change (`app.py`).** Always call the HF download helpers and rely on
`huggingface_hub`'s etag caching — an unchanged file is a cheap metadata check,
a changed file is re-fetched. This also removes the fragile "flatten" hack by
downloading into the correct layout in the first place.

```python
def _ensure_data() -> None:
    """Sync the chroma index + graph pickle from the HF dataset repo.

    Always calls the HF download helpers. huggingface_hub does etag-based
    caching, so an unchanged file is a cheap metadata check and a changed file
    is re-downloaded. A plain restart therefore picks up new data — no
    factory-reboot / storage wipe required.
    """
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        _logger.info("Syncing data from HF dataset (etag-cached)...")
        # Repo layout is chroma/…  -> land it at CHROMA_DIR/chroma.sqlite3 by
        # downloading into CHROMA_DIR's parent (no post-hoc file moves, so etag
        # caching keeps working across restarts).
        snapshot_download(
            repo_id=_HF_DATASET,
            repo_type="dataset",
            local_dir=str(CHROMA_DIR.parent),
            allow_patterns=["chroma/**"],
        )
        hf_hub_download(
            repo_id=_HF_DATASET,
            repo_type="dataset",
            filename="graph/als_graph.pkl",
            local_dir=str(GRAPH_PICKLE_PATH.parent.parent),
        )
        _logger.info("Data sync complete")
    except Exception as e:
        _logger.warning(f"Data sync from HF dataset failed: {e}")
```

**Procedure change.** To ship new data, update the dataset repo, then issue a
**plain restart** — never `factory_reboot=True`:

```python
from huggingface_hub import HfApi
# Re-runs app.py on the SAME built image (no dep reinstall, no version drift);
# _ensure_data() then pulls only the changed file via etag.
HfApi().restart_space("KevinIsCoding/candle-fire")
```

**Infra change (removes the 1.3 GB cold-start entirely).** Enable **persistent
storage** on the Space (Settings → Storage). With persistence the chroma index
survives restarts and `_ensure_data()`'s etag check skips it — only a changed
pickle downloads. Without persistence, storage is ephemeral and every restart
re-downloads ~1.3 GB. Reserve `factory_reboot=True` for genuine image/dependency
changes, and treat it as a change that can expose cold-start regressions.

### Item 3 — Pin runtime-critical dependencies

**Problem.** `requirements.txt` used floors/ranges (`gradio>=6.14.0,<7.0.0`,
`chromadb>=0.5.0`, …). A rebuild can resolve *newer* versions than the
last-known-good, and the Gradio pin drifted away from the README `sdk_version`.

**Code change (`requirements.txt`).** Pin exact versions (source them from a
`pip freeze` on the currently-healthy Space). The Gradio pin MUST equal the
README front-matter `sdk_version`.

```text
# Pin exact versions — regenerate from `pip freeze` on the known-good Space.
# gradio MUST equal README front-matter sdk_version ("6.14.0").
anthropic==<frozen>
gradio==6.14.0
chromadb==<frozen>
networkx==<frozen>
biopython==<frozen>
httpx==<frozen>
python-dotenv==<frozen>
rich==<frozen>
sentence-transformers==<frozen>
openai==<frozen>
rapidfuzz==<frozen>
```

Mirror the same `gradio==6.14.0` pin in `pyproject.toml`, and prefer committing a
lockfile (`uv.lock`) as the source of truth.

**Guardrail (CI / pre-commit)** — fail the build if README `sdk_version` and the
`requirements.txt` Gradio pin disagree:

```bash
#!/usr/bin/env bash
# scripts/check_gradio_pin.sh
set -euo pipefail
readme_ver=$(grep -E '^sdk_version:' README.md | tr -d ' "' | cut -d: -f2)
req_ver=$(grep -E '^gradio==' requirements.txt | cut -d= -f3)
[[ "$readme_ver" == "$req_ver" ]] || {
  echo "Gradio mismatch: README sdk_version=$readme_ver vs requirements gradio==$req_ver" >&2
  exit 1
}
echo "Gradio pin OK ($readme_ver)"
```

### Item 4 — Deploy runbook (`docs/DEPLOY.md`)

candle-fire has **two deploy targets**: **data** → dataset repo
`KevinIsCoding/candle-fire-data`; **code** → Space git `hf/main` (deploy lineage
`hf-clean`). The app downloads data at startup — data is NOT served from the
Space git.

**A. Data-only update (most common)**
1. Rebuild artifacts: `uv run python scripts/build_graph.py` (plus
   `build_index.py` if the corpus changed).
2. Upload to the dataset repo:
   ```bash
   hf upload KevinIsCoding/candle-fire-data data/graph/als_graph.pkl graph/als_graph.pkl --repo-type dataset
   ```
3. Plain restart (NOT factory reboot):
   ```bash
   python -c "from huggingface_hub import HfApi; HfApi().restart_space('KevinIsCoding/candle-fire')"
   ```
4. Verify (section D). No code push needed.

**B. Code update**
1. Make the change on `hf-clean` (or PR into it).
2. Any tracked data file > 10 MB MUST be Git LFS (HF rejects >10 MB non-LFS):
   ```bash
   git lfs track "data/graph/als_graph.json" "data/extracted/entities.jsonl"
   ```
3. Push to the Space (fast-forward; `hf/main` and `origin/hf-clean` stay in lockstep):
   ```bash
   git push origin hf-clean && git push hf hf-clean:main
   ```
4. The Space rebuilds automatically. Verify (section D).

**C. Golden rules**
- Do NOT `factory_reboot` for routine deploys — it wipes storage and rebuilds the
  image, which can expose cold-start regressions. Use a plain restart.
- `ssr_mode=False` must stay in `demo.launch()` (Gradio SSR 503s on this Space).
- `requirements.txt` gradio pin MUST equal README `sdk_version`.

**D. Post-deploy verification (wait out the warmup window).** The app logs
"Running on local URL" *before* HF's proxy is ready; a warmup window (~1–2 min,
longer on a cold 1.3 GB download) returns 503 until it settles. Wait, then probe:

```bash
python - <<'PY'
import time, urllib.request
from huggingface_hub import HfApi
while HfApi().get_space_runtime("KevinIsCoding/candle-fire").stage != "RUNNING":
    time.sleep(15)
time.sleep(120)  # warmup buffer
url, ok = "https://keviniscoding-candle-fire.hf.space/", 0
for _ in range(12):
    try: ok += urllib.request.urlopen(url, timeout=30).status == 200
    except Exception: pass
    time.sleep(6)
print(f"{ok}/12 healthy —", "DEPLOY OK" if ok >= 11 else "INVESTIGATE")
PY
```

Target: ≥ 11/12 HTTP 200. If it stays 503 *after* warmup, read the run logs —
do not reflexively rebuild.

## 10. Lessons Learned

- A running Space can be serving on **warm state** that a cold rebuild will not
  reproduce; forcing a cold rebuild (factory-reboot) is a high-risk action, not
  a routine one.
- **Unpinned/defaulted serving behavior** (Gradio's experimental SSR, unpinned
  Gradio version) is a latent risk that surfaces only on rebuild.
- During an incident, **stop changing things and let the system settle** before
  measuring; rapid iteration on a slow-cold-start service produces misleading
  signals and extends the outage.
- The application layer being healthy does not mean the service is up — the
  **serving/proxy layer** is a distinct failure domain worth checking first
  (the 500/503 body identified the source as the platform, not the app).
