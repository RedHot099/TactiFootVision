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


def inline_svg(path: Path) -> str:
    """The <svg> element alone, for inline use (hover titles work only inline, not in an <img>).

    Matplotlib's global `*{...}` style block is dropped so it does not leak into the page.
    """
    svg = "<svg" + path.read_text().split("<svg", 1)[1]
    return re.sub(r"<style[^>]*>.*?</style>", "", svg, flags=re.S)


def fill(template: str, values: dict[str, str]) -> str:
    missing = set(re.findall(r"\{\{(\w+)\}\}", template)) - values.keys()
    if missing:
        raise ValueError(f"template placeholders without a value: {sorted(missing)}")
    return re.sub(r"\{\{(\w+)\}\}", lambda m: values[m.group(1)], template)


def assets(root: Path) -> dict[str, str]:
    """figures/x.svg -> fig_x (data URI) and svg_x (inline), media/y.jpg -> img_y, media/y.mp4 -> vid_y."""
    values = {}
    for path in sorted((root / "figures").glob("*.svg")):
        values[f"fig_{path.stem}"] = inline(path)
        values[f"svg_{path.stem}"] = inline_svg(path)
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
        digits = {"angle": 0, "def_cone": 0, "opp_3m": 0}
        for key, value in m.get("geometry", {}).items():
            values[f"{name}_g_{key}"] = f"{value:.{digits.get(key, 1)}f}"
    return values


def ball_values(root: Path) -> dict[str, str]:
    meta = json.loads((root / "media/meta.json").read_text())
    values = {}
    for name, m in meta.items():
        if not name.startswith("g_"):
            continue
        for key in ("error_track", "error_top"):
            value = m[key]
            values[f"{name}_{key}"] = "–" if value is None else f"{value:.1f} m"
        values[f"{name}_status"] = m["status"]
    numbers = json.loads((root / "numbers.json").read_text())
    gain = numbers["test_gain_pp"]
    values["test_gain"] = f"+{gain[0]:.1f} pp [+{gain[1]:.1f}, +{gain[2]:.1f}]"
    values["test_track"] = f"{100 * numbers['test_track']['value']:.1f}%"
    values["test_top1"] = f"{100 * numbers['test_top1']['value']:.1f}%"
    values["test_median"] = f"{numbers['test_track_median_m']['value']:.2f} m"
    # base_<split>..., old detector (ball-detection 25): shares and paired gains, one decimal
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    for split, rows in numbers["baseline"].items():
        base = rows["base"]["well_placed"]
        values[f"base_{split}"] = f"{pct(base['value'])}%"
        values[f"base_{split}_ci"] = f"[{pct(base['ci'][0])}, {pct(base['ci'][1])}]"
        values[f"base_far_{split}"] = f"{pct(rows['base']['far_of_placed']['value'])}%"
        for key in ("base_to_dd2c", "base_to_pipeline"):
            d = rows[key]
            values[f"{key}_{split}"] = (
                f"+{pct(d['difference'])} pp [+{pct(d['ci'][0])}, +{pct(d['ci'][1])}]"
            )
    # acc_<row>_<key> and acc_<row>_<key>_ci: games 2+3 thresholds (shares -> %) and percentiles (m)
    for row, keys in numbers["accuracy_games_2_3"].items():
        for key, stat in keys.items():
            scale, unit, digits = (1, " m", 2) if key.endswith("_m") else (100, "%", 1)
            values[f"acc_{row}_{key}"] = f"{scale * stat['value']:.{digits}f}{unit}"
            lo, hi = (scale * c for c in stat["ci"])
            values[f"acc_{row}_{key}_ci"] = f"[{lo:.{digits}f}, {hi:.{digits}f}]"
    return values


def main(name: str) -> None:
    root = OUT / name
    values = assets(root) | {"css": (HERE / "summary.css").read_text()}
    if name == "xg":
        values |= xg_values(root)
    if name == "ball":
        values |= ball_values(root)
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
