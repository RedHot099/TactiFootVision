"""Exercise the real local review page. Synthetic annotations are temporary and never imported."""

import argparse
import json
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

from diagnostic_review import json_text, validate_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--out", type=Path, default=Path("research/v15_diagnostics/browser_check"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/usr/bin/chromium", headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 1000})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(args.url)
        page.wait_for_function("document.getElementById('message').textContent.startsWith('Gotowe:')")
        page.locator("#reviewer").fill("automation-only-do-not-import")
        page.locator("#start").click()
        page.locator("#frame").wait_for()
        page.wait_for_function("document.getElementById('frame').naturalWidth > 0")
        page.screenshot(path=str(args.out / "review.png"), full_page=True)
        packet = page.evaluate("fetch('packet.json').then(r=>r.json())")
        page.locator("summary").click()
        page.locator("#clip").evaluate("video => video.load()")
        page.wait_for_function("document.getElementById('clip').readyState >= 2")
        page.locator("#clip").evaluate("video => { video.currentTime = 3; }")
        page.wait_for_function("Math.abs(document.getElementById('clip').currentTime - 3) < .1 && !document.getElementById('clip').seeking")
        page.locator("#back").click()
        page.locator("#lower").click()
        page.locator("#forward").click()
        page.locator("#upper").click()
        page.locator("#frame").click(position={"x": 350, "y": 200})
        page.locator("#note").fill("AUTOMATION CHECK ONLY; not a human review")
        page.locator("#save").click()
        assert "Oceniono 1 z" in page.locator("#progress").inner_text()
        page.locator("#next").click()
        page.locator("#previous").click()
        assert page.locator("#contact_status").input_value() == "bounded"
        page.reload()
        page.wait_for_function("document.getElementById('message').textContent.startsWith('Gotowe:')")
        page.locator("#reviewer").fill("automation-only-do-not-import")
        page.locator("#start").click()
        assert page.locator("#note").input_value().startswith("AUTOMATION CHECK")
        with tempfile.TemporaryDirectory(prefix="xg-ui-test-") as temp:
            with page.expect_download() as download:
                page.locator("#export").click()
            path = Path(temp) / "synthetic.json"
            download.value.save_as(path)
            payload = json.loads(path.read_text())
            rows = validate_annotations(payload, packet)
            assert len(rows) == 1 and rows.shooter_x.notna().all()
        page.evaluate("localStorage.clear()")
        browser.close()
    if errors:
        raise AssertionError(errors)
    result = {"status": "passed", "checks": ["real decoded image", "interval selection", "shooter point",
        "full clip decoding and seeking", "navigation", "reload persistence", "download and Python validation"], "page_errors": errors,
        "human_annotations_created": 0, "synthetic_export": "temporary file deleted; isolated browser storage cleared"}
    (args.out / "result.json").write_text(json_text(result))
    print(json_text(result))


if __name__ == "__main__":
    main()
