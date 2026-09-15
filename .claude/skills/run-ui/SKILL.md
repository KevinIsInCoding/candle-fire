---
name: run-ui
description: Verify the Candle-Fire Clinical Trials tab in a headless browser using the generic gradio-ui-verify tool with this project's spec. Use after any change to the Clinical Trials tab UI (app.py widgets, trials_query render, the combobox gate/placeholder head-script/CSS) to confirm it renders and behaves correctly — instead of asking the user to restart and eyeball.
---

# run-ui — headless UI verification for the Clinical Trials tab

The launch/drive/assert/screenshot **engine is the independent `gradio-ui-verify` package** — not
vendored here. This project only supplies a **spec** ([`candle_fire_spec.py`](candle_fire_spec.py))
that says how to launch Candle-Fire and what to check. Candle-Fire is, in effect, the worked
example of using that tool.

## Prerequisites (one-time)

```bash
uv pip install gradio-ui-verify        # once published; for local dev: uv pip install -e ../gradio-ui-verify
uv run playwright install chromium     # ~95MB browser download
```

## Run it

```bash
# Launches the app in UI-smoke mode + verifies the Clinical Trials tab + screenshots it
uv run python -m gradio_ui_verify .claude/skills/run-ui/candle_fire_spec.py

# Verify an already-running instance instead of launching one
uv run python -m gradio_ui_verify .claude/skills/run-ui/candle_fire_spec.py --url http://127.0.0.1:7860/
```

**Run it in the background** and read the output — the smoke launch takes ~25s and Python buffers
when piped. Exit code is 0 if all checks pass, non-zero otherwise (CI-usable). It always writes a
full-page screenshot to `docs/ui-mocks/` — **look at it**; a blank frame is a failed launch.

## What the spec checks

facility & city comboboxes present · in-box placeholder set on both · 3-char gate (list hidden at
2 chars, shown at 3) · picked value is clean (no "· N" suffix leak) · "Recruitment status" label
on one line · a search returns a result panel. Edit [`candle_fire_spec.py`](candle_fire_spec.py)
to add checks (use the `gradio_ui_verify.checks` helpers, or drive the Playwright `page` directly).

## App-side support (in this repo)

- `CANDLE_UI_SMOKE=1` (app.py) skips heavy startup loads so the UI boots fast; the spec sets it.
- Combobox placeholder + 3-char gate are wired via `gr.Blocks(head="<script>…")` + a
  MutationObserver — Gradio ignored `js=`/`demo.load(js=)`, and the tab renders lazily.

See `docs/ui-verification-todo.md` for the fuller design→mock→verify plan (Phases 4–5 not built).
The generic tool and its own copy of this skill live in the `gradio-ui-verify` repo.
