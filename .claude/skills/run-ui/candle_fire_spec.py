"""UI verification spec for Candle-Fire's Clinical Trials tab.

Consumed by the generic `gradio-ui-verify` tool (an independent package):
    python -m gradio_ui_verify .claude/skills/run-ui/candle_fire_spec.py

Requires `gradio-ui-verify` installed (see SKILL.md). This file is the ONLY project-specific
part — the launch/drive/assert/screenshot engine lives in the package.
"""
from gradio_ui_verify import checks as c

# Launch the app in UI-smoke mode (CANDLE_UI_SMOKE skips heavy model loads; see app.py) so the
# interface boots fast for layout/behavior checks.
LAUNCH = {
    "cmd": ["uv", "run", "python", "app.py"],
    "port": 7899,
    "env": {"CANDLE_UI_SMOKE": "1", "GRADIO_SERVER_PORT": "7899"},
}
SHOT = "docs/ui-mocks/clinical-trials-actual.png"


def checks(page, result) -> None:
    # Open the Clinical Trials tab, then wait for its lazily-rendered combobox.
    page.get_by_role("tab", name="Clinical Trials").click()
    page.wait_for_selector("#facility_combo input", timeout=15000)
    page.wait_for_timeout(1000)  # let the head-script MutationObserver wire the combobox

    print("Clinical Trials tab checks:")
    c.present(page, result, "#facility_combo input", "facility combobox present")
    c.present(page, result, "#city_combo input", "city combobox present")
    c.attr_nonempty(page, result, "#facility_combo input", "placeholder", "facility placeholder set")
    c.attr_nonempty(page, result, "#city_combo input", "placeholder", "city placeholder set")
    c.class_gate_on_input(
        page, result, "#facility_combo", "#facility_combo input", "ac-hide",
        below="ma", at="mas", label="3-char gate (hidden at 2, shown at 3)",
    )
    fac = page.query_selector("#facility_combo input")
    fac.fill("mass gen")
    page.wait_for_timeout(400)
    opt = page.query_selector("#facility_combo li, #facility_combo [role='option']")
    if opt:
        opt.click()
        page.wait_for_timeout(200)
    c.value_clean(page, result, "#facility_combo input", ("·", "site"), "picked value is clean")
    c.one_line(page, result, "Recruitment status", 28, "‘Recruitment status’ label on one line")

    page.get_by_role("button", name="Search trials").click()
    page.wait_for_timeout(1500)
    c.text_present(page, result, ["Eligibility", "trial(s) matched", "No "], "search returns a result panel")
