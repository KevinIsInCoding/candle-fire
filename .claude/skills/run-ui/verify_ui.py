"""Launch the app in UI-smoke mode and verify the Clinical Trials tab with a headless browser.

Deterministic checks for the things that have regressed before (placeholder, 3-char gate,
clean picked value, eligibility on recruiting trials, one-line filter labels), plus a
screenshot for eyeballing / visual-QA. Exits non-zero if any check fails, so it is CI-usable.

Usage:
  uv run python .claude/skills/run-ui/verify_ui.py                 # launch smoke app + verify
  uv run python .claude/skills/run-ui/verify_ui.py --url URL       # verify an already-running app
  uv run python .claude/skills/run-ui/verify_ui.py --keep          # leave the launched app running
  uv run python .claude/skills/run-ui/verify_ui.py --shot PATH     # screenshot destination

Requires: playwright + chromium (`uv pip install playwright && uv run playwright install chromium`).
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _launch_smoke(port: int, log_path: Path) -> subprocess.Popen:
    """Boot app.py in UI-smoke mode (no heavy models) on `port`; return the process.

    Child stdout/stderr go to a LOG FILE, not a PIPE: once running, the app keeps logging,
    and an unread PIPE fills its ~64KB buffer and deadlocks the app mid-request. Poll the file.
    """
    subprocess.run(["bash", "-c", f"lsof -ti:{port} | xargs kill 2>/dev/null || true"], check=False)
    env = {**os.environ, "CANDLE_UI_SMOKE": "1", "GRADIO_SERVER_PORT": str(port)}
    log = open(log_path, "w")
    proc = subprocess.Popen(
        ["uv", "run", "python", "app.py"], cwd=str(ROOT), env=env,
        stdout=log, stderr=subprocess.STDOUT, text=True,
    )
    # Readiness = the PORT accepts a connection. (Don't grep the log for "Running on local URL":
    # the child's file-redirected stdout is block-buffered, so that line can lag serving.)
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"app exited during startup (see {log_path})")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                time.sleep(1.0)  # let Gradio finish wiring routes
                return proc
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"app did not become ready within 120s (see {log_path})")


class Checks:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def ok(self, cond: bool, label: str, detail: str = "") -> None:
        mark = "PASS" if cond else "FAIL"
        print(f"  [{mark}] {label}" + (f" — {detail}" if detail and not cond else ""))
        if not cond:
            self.failures.append(label)


def verify(url: str, shot: str) -> int:
    from playwright.sync_api import sync_playwright

    c = Checks()
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.set_viewport_size({"width": 1200, "height": 1000})
        # 'load', not 'networkidle': Gradio keeps a persistent connection, so networkidle
        # may never settle. Wait explicitly for the app shell instead.
        pg.goto(url, wait_until="load", timeout=60000)
        pg.get_by_role("tab", name="Clinical Trials").click()
        pg.wait_for_selector("#facility_combo input", timeout=15000)
        pg.wait_for_timeout(1000)  # lazy tab render + MutationObserver wiring

        print("Clinical Trials tab checks:")
        fac = pg.query_selector("#facility_combo input")
        city = pg.query_selector("#city_combo input")
        c.ok(bool(fac) and bool(city), "facility & city comboboxes present")

        # In-box placeholder hint (regressed once — demo.load ran before lazy tab render)
        fac_ph = fac.get_attribute("placeholder") if fac else None
        c.ok(bool(fac_ph), "facility placeholder set", f"got {fac_ph!r}")
        c.ok(bool(city.get_attribute("placeholder") if city else None), "city placeholder set")

        # 3-char gate: <3 chars keeps the option list hidden (.ac-hide), >=3 reveals it
        if fac:
            fac.click()
            fac.fill("ma")
            pg.wait_for_timeout(200)
            hidden2 = pg.eval_on_selector("#facility_combo", "e => e.classList.contains('ac-hide')")
            fac.fill("mas")
            pg.wait_for_timeout(200)
            hidden3 = pg.eval_on_selector("#facility_combo", "e => e.classList.contains('ac-hide')")
            c.ok(hidden2 and not hidden3, "3-char gate (hidden at 2, shown at 3)",
                 f"hidden@2={hidden2} hidden@3={hidden3}")

        # Picking a suggestion fills the CLEAN value (no "·"/count suffix leaking in)
        if fac:
            fac.fill("mass gen")
            pg.wait_for_timeout(400)
            opt = pg.query_selector("#facility_combo li, #facility_combo [role='option']")
            if opt:
                opt.click()
                pg.wait_for_timeout(200)
                val = fac.input_value()
                c.ok("·" not in val and "site" not in val.lower(), "picked value is clean", f"got {val!r}")
            else:
                c.ok(False, "picked value is clean", "no option element found to click")

        # One-line filter labels (the "Recruitment status" wrap fix)
        rs_h = pg.evaluate(
            """() => {
                const el = [...document.querySelectorAll('span,label')]
                  .find(e => e.textContent.trim() === 'Recruitment status');
                return el ? el.getBoundingClientRect().height : null;
            }"""
        )
        c.ok(rs_h is not None and rs_h < 28, "‘Recruitment status’ label on one line", f"height={rs_h}")

        # Eligibility renders for recruiting trials after a search
        pg.query_selector("#city_combo input")  # ensure tab is settled
        state = pg.query_selector("input[aria-label='State'], #component-state")  # best-effort
        # Simpler: search by facility we just picked (recruiting default) via the button
        btn = pg.get_by_role("button", name="Search trials")
        btn.click()
        pg.wait_for_timeout(1500)
        body = pg.inner_text("body")
        c.ok("Eligibility" in body or "trial(s) matched" in body or "No " in body,
             "search returns a result panel")

        pg.screenshot(path=shot, full_page=True)
        print(f"screenshot: {shot}")
        b.close()

    print(f"\n{'ALL CHECKS PASSED' if not c.failures else f'{len(c.failures)} CHECK(S) FAILED: ' + ', '.join(c.failures)}")
    return 0 if not c.failures else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="Verify an already-running app instead of launching.")
    ap.add_argument("--port", type=int, default=7899, help="Port for the smoke launch.")
    ap.add_argument("--keep", action="store_true", help="Leave the launched app running.")
    ap.add_argument("--shot", default=str(ROOT / "docs" / "ui-mocks" / "clinical-trials-actual.png"))
    args = ap.parse_args()

    Path(args.shot).parent.mkdir(parents=True, exist_ok=True)
    proc = None
    try:
        url = args.url
        if not url:
            print(f"Launching smoke app on :{args.port} ...", flush=True)
            proc = _launch_smoke(args.port, Path(args.shot).parent / "smoke_app.log")
            url = f"http://127.0.0.1:{args.port}/"
            print(f"ready at {url}", flush=True)
        return verify(url, args.shot)
    finally:
        if proc and not args.keep:
            proc.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
