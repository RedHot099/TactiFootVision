"""Locate the static broadcast overlay (scoreboard / clock / channel logo) from data.

    ./run.sh python scripts/find_overlay_region.py --out configs

Between t0 and t-1600 ms the camera pans and every player moves, so pixels that stay
identical are burned-in graphics. Raw |t0 - t(-1600)| is not enough on its own:
plain grass also barely changes, simply because it has no texture to change. The
discriminative quantity is therefore *temporal change per unit of spatial detail* -
an overlay is sharp (text, logos, box edges) yet frozen, grass is smooth and calm,
crowd and players are both detailed and moving.

Components are kept only inside the outer margin of the frame, since broadcast
graphics are anchored to the edges while a coincidentally still patch of pitch is not.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from xgpoc.data.index import load_index

from xgpoc.paths import INDEX as _INDEX  # noqa: E402
INDEX = str(_INDEX)


def pair_stats(paths: tuple[str, str]) -> tuple[np.ndarray, np.ndarray] | None:
    """(temporal abs diff, spatial detail) maps for one shot."""
    a, b = (cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths)
    if a is None or b is None or a.shape != b.shape:
        return None
    detail = np.abs(cv2.Laplacian(a, cv2.CV_32F, ksize=3))
    return cv2.absdiff(a, b).astype(np.float32), cv2.blur(detail, (9, 9))


def anchored_components(mask: np.ndarray, margin: float = 0.22, min_area: int = 300) -> np.ndarray:
    """Keep blobs that live in the outer `margin` band of the frame."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    h, w = mask.shape
    keep = np.zeros_like(mask, bool)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < min_area:
            continue
        if y < margin * h or y + ch > (1 - margin) * h or x < margin * w or x + cw > (1 - margin) * w:
            keep |= lab == i
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX)
    ap.add_argument("--out", type=Path, default=Path("configs"))
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--quantile", type=float, default=0.03,
                    help="fraction of the most 'frozen yet detailed' pixels taken as overlay")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    df = load_index(Path(a.index)).sample(a.n, random_state=0)
    with ThreadPoolExecutor(16) as ex:
        res = [r for r in ex.map(pair_stats, zip(df.frame_path, df["frame_path_-1600"])) if r is not None]
    temporal = np.mean([r[0] for r in res], 0)
    detail = np.mean([r[1] for r in res], 0)
    score = temporal / (detail + 1e-3)           # low = frozen despite being sharp
    print(f"{len(res)} usable pairs, frame {temporal.shape}")

    thr = float(np.quantile(score, a.quantile))
    raw = (score < thr) & (detail > np.median(detail))
    mask = cv2.dilate(anchored_components(raw).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    ys, xs = np.where(mask)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    rep = {
        "n_pairs": len(res), "score_threshold": thr, "quantile": a.quantile,
        "masked_pixels_pct": round(100 * float(mask.mean()), 3),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if mask.any() else None,
        "components": [{"xyxy": [int(x), int(y), int(x + w), int(y + h)], "area": int(ar)}
                       for x, y, w, h, ar in sorted(stats[1:].tolist(), key=lambda s: -s[4])[:8]],
        "mean_temporal_diff_inside": round(float(temporal[mask].mean()), 2) if mask.any() else None,
        "mean_temporal_diff_outside": round(float(temporal[~mask].mean()), 2),
    }
    # A second, stricter mask keeping only the graphics *boxes*: components that sit
    # wholly in the top quarter and are tall enough to be a widget rather than a 1-2 px
    # encoding border. The wide mask also contains thin strips along the frame edges,
    # and those strips carry pitch pixels - a probe trained on them would measure camera
    # framing, not the scoreboard, and would attribute the wrong cause to the result.
    graphics = np.zeros_like(mask)
    for i in range(1, n):
        x, y, cw, ch, ar = stats[i]
        if ch >= 20 and ar >= 1500 and y + ch < 0.25 * mask.shape[0]:
            graphics |= lab == i
    rep["graphics_components"] = int(cv2.connectedComponents(graphics.astype(np.uint8))[0] - 1)
    rep["graphics_pixels_pct"] = round(100 * float(graphics.mean()), 3)

    (a.out / "overlay_mask.json").write_text(json.dumps(rep, indent=2))
    cv2.imwrite(str(a.out / "overlay_mask.png"), (mask * 255).astype(np.uint8))
    cv2.imwrite(str(a.out / "graphics_mask.png"), (graphics * 255).astype(np.uint8))

    heat = cv2.applyColorMap(cv2.normalize(np.clip(score, 0, np.quantile(score, 0.99)), None, 0, 255,
                                           cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_INFERNO)
    sample = cv2.imread(df.frame_path.iloc[0])
    tinted = sample.copy()
    tinted[mask] = (0, 0, 255)
    cv2.imwrite(str(a.out / "overlay_diagnostic.jpg"),
                np.vstack([heat, cv2.addWeighted(sample, 0.45, tinted, 0.55, 0)]))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
