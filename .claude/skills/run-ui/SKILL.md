---
name: run-ui
description: Launch the Candle-Fire Gradio app in a headless browser and verify the Clinical Trials tab against deterministic UI checks (plus a screenshot). Use after any change to the Clinical Trials tab UI (app.py widgets, trials_query render, the combobox gate/placeholder JS/CSS) to confirm it renders and behaves correctly — instead of asking the user to restart and eyeball.
---

# run-ui — headless UI verification for the Clinical Trials tab

Gradio renders in the browser, so Python tests can't catch layout/JS regressions (wrapped
labels, a placeholder that never attaches, a broken 3-char gate, an eligibility block that
doesn't show). This skill drives the real app with Playwright + Chromium and asserts the
behaviors that have broken before.

## Prerequisites (one-time)

```bash
uv pip install playwright
uv run playwright install chromium   # ~95MB; confirmed to work in this environment
```

## Run it

```bash
# Launch a fast UI-smoke instance (no heavy models) + verify + screenshot
uv run python .claude/skills/run-ui/verify_ui.py

# Verify an already-running instance instead of launching one
uv run python .claude/skills/run-ui/verify_ui.py --url http://127.0.0.1:7860/

# Options
#   --port N     port for the smoke launch (default 7899)
#   --shot PATH  screenshot destination (default docs/ui-mocks/clinical-trials-actual.png)
#   --keep       leave the launched app running
```

Exit code is **0 if all checks pass, non-zero otherwise** (CI-usable). Each check prints
`[PASS]`/`[FAIL]`. Always writes a full-page screenshot — **look at it**; a blank frame is a
failure to launch.

**Run it in the background** (`run_in_background`) and read the output file — the smoke launch
takes ~25s and Python buffers when piped. Don't foreground it behind a `| grep`.

## What it checks (Phase 3)

- facility & city comboboxes present
- in-box placeholder set on both (regressed once — see "Gotchas")
- 3-char gate: option list hidden at 2 chars, shown at 3
- picking a suggestion fills the CLEAN value (no "· N" count suffix leaking in)
- "Recruitment status" label renders on one line
- a search returns a result panel (eligibility/results/empty-hint)

Add a check by extending the `Checks` block in `verify_ui.py`. Target the comboboxes by
`#facility_combo` / `#city_combo` (their `elem_id`s); other widgets by role/text.

## How it works

- **UI-smoke mode** (`CANDLE_UI_SMOKE=1`, set by the launcher): `app.py` skips the heavy
  startup loads (cross-encoder, ChromaDB, graph, Anthropic client) and renders the interface
  with just the trials list. Boot ~25s (dominated by torch/chromadb imports) vs ~60s full.
  Query features are inert in this mode; layout/search-over-trials still work.
- **Readiness = the port accepts a connection**, not a log line (the child's file-redirected
  stdout is block-buffered, so "Running on local URL" can lag serving).

## Gotchas learned building this (don't relearn them)

- **Gradio `js=` / `demo.load(js=...)` did NOT execute** in this version. Inject browser JS via
  `gr.Blocks(head="<script>…</script>")` instead — a real `<head>` script runs directly.
- The Clinical Trials tab **renders lazily** (inputs exist only after the tab is opened), so the
  head script uses a **MutationObserver** to wire the combobox once it appears.
- Don't `stdout=PIPE` a long-running child and stop reading it — the pipe fills (~64KB) and
  deadlocks the app mid-request. Log to a file.
- Gradio keeps a persistent connection, so `wait_until="networkidle"` may never settle — use
  `"load"` and wait for `#facility_combo input`.

## Not yet built (see docs/ui-verification-todo.md)

Phase 4–5: describe → HTML mock → reference PNG → a `ui-reviewer` subagent that compares the
screenshot to the mock for subjective design fidelity. This skill is the deterministic gate.
