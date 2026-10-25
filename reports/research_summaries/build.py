"""Assemble a research summary into one self-contained HTML file and a PDF.

Run: uv run python reports/research_summaries/build.py xg
The template is <name>_summary.html next to this file; {{key}} placeholders are
filled from the figures, media and numbers under outputs/research_summaries/<name>/.
Images and the clip are inlined (base64) so the HTML travels as one file; the PDF
is printed by headless Chromium, where the clip is replaced by its poster frame.
"""

import base64
import json
import re
import subprocess
import sys
from pathlib import Path

from sources import OUT

HERE = Path(__file__).parent
MIME = {
    ".svg": "image/svg+xml",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".mp4": "video/mp4",
}


def inline(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{MIME[path.suffix]};base64,{data}"


def fill(template: str, values: dict[str, str]) -> str:
    missing = set(re.findall(r"\{\{(\w+)\}\}", template)) - values.keys()
    if missing:
        raise ValueError(f"template placeholders without a value: {sorted(missing)}")
    return re.sub(r"\{\{(\w+)\}\}", lambda m: values[m.group(1)], template)


def assets(root: Path) -> dict[str, str]:
    """figures/x.svg -> fig_x, media/y.jpg -> img_y, media/y.mp4 -> vid_y."""
    values = {}
    for path in sorted((root / "figures").glob("*.svg")):
        values[f"fig_{path.stem}"] = inline(path)
    for path in sorted((root / "media").glob("*")):
        prefix = {".jpg": "img", ".png": "img", ".mp4": "vid"}.get(path.suffix)
        if prefix:
            values[f"{prefix}_{path.stem}"] = inline(path)
    return values


def xg_values(root: Path) -> dict[str, str]:
    meta = json.loads((root / "media/meta.json").read_text())
    values = {}
    for name, m in meta.items():
        for key, value in m["xg"].items():
            values[f"{name}_{key}"] = f"{value:.2f}"
        values[f"{name}_outcome"] = "Goal" if m["goal"] else "No goal"
        values[f"{name}_caption"] = m["caption"]
        values[f"{name}_match"] = m["match"].split("/")[-1]
    return values


def main(name: str) -> None:
    root = OUT / name
    values = assets(root)
    if name == "xg":
        values |= xg_values(root)
    html = fill((HERE / f"{name}_summary.html").read_text(), values)
    out_html = root / f"{name}_summary.html"
    out_html.write_text(html)
    out_pdf = root / f"{name}_summary.pdf"
    subprocess.run(
        [
            "chromium",
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={out_pdf}",
            out_html.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    print(out_html, out_pdf)


if __name__ == "__main__":
    main(sys.argv[1])
