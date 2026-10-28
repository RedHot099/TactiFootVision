"""Masked variants of a shot's t0 frame and their out-of-fold G-combo scores (xg-dino-info ticket 14; spec-info §(d),
council C3.5 and the notes after ticket 09; protocol `research/xgdino_info_protocol.json`, `frozen.masking`).

Variants of one shot (`shot_variants`), all on G-combo's model input (448 x 252 after the overlay mask):

    unmasked          the frame as G-combo saw it; its score must reproduce the stored OOF `z_g_combo`
    region            one stored region of ticket 09 (shooter, goalkeeper, ball, goal, players, background) and
                      `shooter+ball` (the union of the stored shooter and ball masks, present when both are)
    placebo           the stored placebo draws of ticket 09; for `shooter+ball` the draws of
                      `regions.placebos(union, allowed, regions.rng_for(shot_id, "shooter+ball"))`, with `allowed`
                      rebuilt from the replayed roles exactly as ticket 09 (`allowed_patches`)
    keep              keep-only variants as mask complements: keep_shooter_ball = ~(shooter | ball),
                      keep_persons = ~players, keep_goal = ~goal; built from the named regions that are present

Absent regions get no variant (ticket 15 reads them as the unmasked frame, flagged). Every masked variant is scored
with both fills (`FILLS`):

    mean   every masked pixel set to the frame's mean colour (mean of the model input after the overlay mask, rounded
           to uint8, as ticket 09's gallery preview)
    blur   every masked pixel set to the same frame blurred with a Gaussian of sigma `BLUR_SIGMA` px (2 patches), so a
           player-sized object is a smear of its surroundings

Pixels the overlay mask already blanks keep the overlay grey: masking them changes nothing (`regions.overlay_share`).

Scoring (`FoldScorer`): every frame goes through the shared frozen trunk once and through the tails of both seeds of
the shot's own held-out fold; `z` = mean over the two seeds of logit / T (per-member temperature), the scale of the
stored `z_g_combo`; `pool` = each seed's 5 x 1024 attention-pool vector (`g_bundle.g_outputs`).
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn

from xgpoc.dino import regions as rg
from xgpoc.probe.scoring import GMember, SharedScorer, ensemble_logits

FILLS = ("mean", "blur")
BLUR_SIGMA = 2.0 * rg.PATCH
UNION = "shooter+ball"
UNION_PARTS = ("shooter", "ball")
KEEP = {"keep_shooter_ball": ("shooter", "ball"), "keep_persons": ("players",), "keep_goal": ("goal",)}
MASKED_REGIONS = (*rg.REGIONS, UNION)


# ------------------------------------------------------------------ variants
@dataclass(frozen=True)
class Variant:
    """One input of a shot. `mask`: the patch mask to fill (None for the unmasked frame); `parts`: the regions a keep
    variant was built from."""

    name: str                       # "unmasked", "<region>", "<region>|placebo<k>", "<keep variant>"
    kind: str                       # unmasked / region / placebo / keep
    region: str                     # "" for the unmasked frame
    draw: int                       # placebo draw, else 0
    dx: int
    dy: int
    mask: np.ndarray | None
    parts: str = ""


def allowed_patches(r, overlay: np.ndarray) -> np.ndarray:
    """Ticket 09's placebo domain of one t0 row (`regions.shot_regions`): patches on the projected pitch, outside the
    overlay and off every patch a person box, the own ball or the goal shape touches (undilated)."""
    pitch, exclusion = rg.empty(), rg.empty()
    if r.frame is None:
        return pitch
    for box in r.boxes():
        exclusion |= rg.box_patches(box)
    if r.own_ball is not None:
        exclusion |= rg.box_patches(r.own_ball)
    H = r.frame.homography
    if H is not None and r.ok and r.s.attack_right is not None:
        polygon, _ = rg.goal_polygon(H, r.s.attack_right)
        if polygon is not None:
            exclusion |= rg.polygon_patches(polygon)
    if H is not None:
        pitch = rg.pitch_patches(H)
    return pitch & ~overlay & ~exclusion


def union_placebos(shot_id: str, union: np.ndarray, allowed: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    """The `shooter+ball` placebo shifts (protocol `frozen.masking["shooter+ball"]`) and the number of feasible ones."""
    return rg.placebos(union, allowed, rg.rng_for(shot_id, UNION))


def shot_variants(shot_id: str, stored: dict[tuple[str, str, int], tuple[np.ndarray, int, int]],
                  allowed: np.ndarray) -> tuple[list[Variant], dict]:
    """(variants, the shooter+ball placebo record) of one shot. `stored`: (region, kind, draw) -> (mask, dx, dy) of
    ticket 09's region_masks rows for this shot."""
    out = [Variant("unmasked", "unmasked", "", 0, 0, 0, None)]
    regions = {region: m for (region, kind, _), (m, _, _) in stored.items() if kind == "region"}
    for region in rg.REGIONS:
        if region not in regions:
            continue
        out.append(Variant(region, "region", region, 0, 0, 0, regions[region]))
        for draw in range(rg.N_PLACEBO):
            if (region, "placebo", draw) in stored:
                m, dx, dy = stored[(region, "placebo", draw)]
                out.append(Variant(f"{region}|placebo{draw}", "placebo", region, draw, dx, dy, m))
    record = {"shot_id": shot_id, "present": False, "n_patches": 0, "placebo_status": "region_absent", "n_feasible": 0}
    if all(p in regions for p in UNION_PARTS):
        union = regions["shooter"] | regions["ball"]
        shifts, n = union_placebos(shot_id, union, allowed)
        record |= {"present": True, "n_patches": int(union.sum()), "n_feasible": n,
                   "placebo_status": "ok" if shifts else "no_valid_placebo"}
        out.append(Variant(UNION, "region", UNION, 0, 0, 0, union))
        out += [Variant(f"{UNION}|placebo{k}", "placebo", UNION, k, dx, dy, rg.shift(union, dx, dy))
                for k, (dx, dy) in enumerate(shifts)]
    for name, parts in KEEP.items():
        have = [p for p in parts if p in regions]
        if have:
            kept = np.logical_or.reduce([regions[p] for p in have])
            out.append(Variant(name, "keep", name, 0, 0, 0, ~kept, "+".join(have)))
    return out, record


# ------------------------------------------------------------------ fills
def patch_pixels(mask: np.ndarray) -> np.ndarray:
    """Input-resolution boolean image `[252, 448]` of a patch mask `[18, 32]`."""
    return np.repeat(np.repeat(mask, rg.PATCH, axis=0), rg.PATCH, axis=1)


def fill_sources(img: np.ndarray) -> dict[str, np.ndarray]:
    """Per fill the image whose pixels replace the masked ones. `img`: model input (H, W, 3) uint8 after the overlay."""
    mean = np.broadcast_to(img.reshape(-1, 3).mean(axis=0).round().astype(np.uint8), img.shape)
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=BLUR_SIGMA, sigmaY=BLUR_SIGMA, borderType=cv2.BORDER_REFLECT)
    return {"mean": mean, "blur": blur}


def apply(img: np.ndarray, mask: np.ndarray | None, source: np.ndarray | None, overlay_pixels: np.ndarray) -> np.ndarray:
    """`img` with the pixels of `mask` (outside the overlay) taken from `source`; a copy, or `img` itself if no mask."""
    if mask is None:
        return img
    pixels = patch_pixels(mask) & ~overlay_pixels
    out = img.copy()
    out[pixels] = source[pixels]
    return out


# ------------------------------------------------------------------ scoring
class FoldScorer:
    """`SharedScorer` with every seed of a fold: a batch of one fold's frames goes through the trunk once and through
    the tail of each registered seed of that fold."""

    def __init__(self, pretrained: nn.Module, n_trained: int, device: str):
        self.scorer = SharedScorer(pretrained.eval(), n_trained, device)
        self.by_fold: dict[int, list[GMember]] = {}

    def add(self, fold: int, seed: int, temperature: float, model: nn.Module) -> None:
        """Register a fold model (frozen part checked against the shared trunk); a (fold, seed) only once."""
        if any(m.seed == seed for m in self.by_fold.get(fold, [])):
            raise ValueError(f"fold {fold} seed {seed} is registered twice")
        member = self.scorer.add("g_combo", fold, seed, float(temperature), model)
        self.by_fold.setdefault(fold, []).append(member)
        self.by_fold[fold].sort(key=lambda m: m.seed)

    @property
    def seeds(self) -> dict[int, list[int]]:
        return {f: [m.seed for m in ms] for f, ms in sorted(self.by_fold.items())}

    @torch.no_grad()
    def score(self, images: torch.Tensor, fold: int) -> dict[str, np.ndarray]:
        """z [B] (mean of logit / T over the fold's seeds, float64), logit [B, S] (raw, per seed), on_target [B]
        (mean raw), pool [B, S, 5 D] float32; seeds in ascending order."""
        members = self.by_fold.get(fold)
        if not members:
            raise ValueError(f"no model for fold {fold}")
        scores, _ = self.scorer.score(images, members, embed=members, frozen=False)
        z, _, on_target = ensemble_logits(scores, members)
        return {"z": z, "logit": np.stack([s.xg for s in scores], 1).astype(np.float64), "on_target": on_target,
                "pool": np.stack([s.embedding for s in scores], 1)}
