# TODO — UI design→mock→implement→verify loop

> **Status (in progress):** Phases 0–3 + 6 built and green — `.claude/skills/run-ui/` launches
> the app in UI-smoke mode, drives the Clinical Trials tab in headless Chromium, and asserts the
> key behaviors (all PASS). Building it caught a real bug: the combobox placeholder + 3-char gate
> **never ran** (Gradio ignored `js=`/`demo.load(js=)`) — fixed by injecting the script via
> `gr.Blocks(head=...)`. Still to do: Phase 1 (<5s smoke boot — currently ~25s), Phases 4–5
> (design-mock front half + visual-QA agent).

**Gap:** UI changes can't be verified from the agent's environment (no browser tooling), so
fixes ship "blind" and rely on the user restarting the app and eyeballing. We also want the
front half: describe a change → see a visual mock → iterate → implement → auto-verify against
that mock.

**Two reframes that make this real (read first):**
1. **The "mock" is HTML, not a hand-painted PNG.** Claude can't render a raster image from a
   description, but it can generate an HTML mock that renders visually and *exports* to PNG.
   The PNG is the **reference snapshot** the automated test diffs the real app against.
2. **Gradio owns the DOM.** The Clinical Trials tab is `gr.Dropdown`/`gr.Row`/etc.; Gradio
   generates the markup. So "mock → code" here means **mock → Gradio components + targeted
   CSS/JS hooks**, not arbitrary DOM/JS. Fidelity is "close," not pixel-perfect. (Pixel control
   would require moving the UI to a raw-HTML/React custom component — a much bigger change,
   out of scope unless we decide otherwise.)

Sequence deliberately: prove the risky/automated half (Phases 0–3) before investing in the
design front half (Phases 4–5). Stop and report if Phase 0 fails.

---

## Phase 0 — Feasibility spike (DO FIRST; may kill the automated half)
- [ ] Confirm Playwright + Chromium can install in the agent environment
      (`uv pip install playwright && playwright install chromium`); note if network-restricted.
- [ ] Boot the app on a test port and confirm a headless Chromium can reach it
      (navigate to `http://127.0.0.1:<port>`, read the page title).
- [ ] **If blocked:** stop; switch to the fallback (user runs the browser, agent drives via a
      script the user executes). Record the decision here.

## Phase 1 — Fast "UI smoke" launch mode
- [ ] Add a lightweight launch path that renders the Blocks WITHOUT the heavy startup loads
      (cross-encoder, ~31k-chunk ChromaDB, graph) — layout checks don't need them.
      e.g. `CANDLE_UI_SMOKE=1` stubs `_collection`/`_graph`/`_cross_encoder` and loads a tiny
      trials subset for the combobox vocab.
- [ ] Target cold-start < ~5s in smoke mode (vs ~60s full).
- [ ] Verify the Clinical Trials tab renders with the smoke subset.

## Phase 2 — Playwright driver
- [ ] Script: launch app (smoke mode, test port, background) → wait for ready → open the
      🏥 Clinical Trials tab.
- [ ] Drive the combobox: type into facility/city, read the suggestion list, pick an option,
      run a search.
- [ ] Capture a screenshot of the tab to a known path.

## Phase 3 — Deterministic assertions (the reliable gate; the things that actually broke)
- [ ] `Recruitment status` label renders on ONE line (measure label box height / line count).
- [ ] Dropdown chevron does NOT overlap the value text (compare bounding boxes).
- [ ] 3-char gate: 2 chars → option list hidden; 3 chars → visible.
- [ ] Placeholder shows when empty and clears on typing.
- [ ] Picking a suggestion fills the box with the CLEAN value (no count suffix).
- [ ] Defaults are Interventional + Recruiting; empty-result hint appears when a location has
      only trials outside the active filters.
- [ ] Eligibility section shows for recruiting trials, hidden for completed.
- [ ] Headline status-bar counts are present and non-hardcoded (match loaded data).
- [ ] Assertions exit non-zero on failure (CI-usable).

## Phase 4 — Design front half: describe → HTML mock → iterate → reference PNG
- [ ] Use the `design` skill (multi-artboard canvas) or `frontend-design` to turn a text
      description into an HTML mock of the target tab/component.
- [ ] Iterate on the HTML mock (edit + re-render) until approved.
- [ ] Export the approved mock to a **reference PNG** stored alongside the test
      (e.g. `docs/ui-mocks/clinical-trials.png`).

## Phase 5 — Close the loop: implement in Gradio + visual-QA agent
- [ ] Translate the approved mock into Gradio components + CSS/JS hooks.
- [ ] A `ui-reviewer` subagent takes the Phase-2 screenshot + the Phase-4 reference PNG + a
      requirements checklist and reports pass/fail with reasons (fidelity, spacing, alignment).
- [ ] Loop: implement → screenshot → compare to reference → adjust, until it matches.
- [ ] (Optional) pixel-diff the screenshot vs reference for regression alarms.

## Phase 6 — Package as a project `run-ui` skill
- [ ] Commit as `.claude/skills/run-ui/` (SKILL.md + driver + smoke-mode notes) so any agent
      invokes it the same way; document how to add an assertion and where artifacts land.
- [ ] Wire into the PR/deploy flow: no UI change ships without this passing.

## Notes / open questions
- Deterministic DOM assertions (Phase 3) are the trustworthy regression gate; the visual-QA
  agent (Phase 5) is for subjective design judgment. Use both; don't gate on the fuzzy one.
- Playwright install feasibility (Phase 0) is the single biggest unknown — everything
  automated depends on it.

## Definition of done
- [ ] An agent can run one command to launch + assert + screenshot the Clinical Trials tab.
- [ ] The regressions in Phase 3 are covered by assertions that fail loudly.
- [ ] A described UI change can be mocked, approved as a reference PNG, implemented, and
      auto-checked against that reference.
