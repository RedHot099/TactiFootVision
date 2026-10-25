"""What the geometry model sees for one shot: the measured inputs, drawn on the frame and from above.

Everything comes from the research cache the stacked model was fitted on (E3 detections at the
shot frame, feet_t0 shot location), so the pictures show the model's actual inputs.
"""

from dataclasses import dataclass

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc, Circle, Polygon, Rectangle
from PIL import Image, ImageDraw, ImageFont
from sources import XD

LENGTH, WIDTH, GOAL_HALF = 105.0, 68.0, 3.66
FONT = "/usr/share/fonts/noto/NotoSans-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/noto/NotoSans-Bold.ttf"
SHOOTER = (42, 120, 214)
DEFENDER = (235, 104, 52)
KEEPER = (237, 161, 0)
CONE = (255, 255, 255)


@dataclass
class ShotGeometry:
    homography: np.ndarray  # image -> pitch metres
    pitch_xy: np.ndarray  # (N, 2) feet of every detected person
    boxes: np.ndarray  # (N, 4) xyxy
    teams: np.ndarray  # (N,)
    keeper: int
    shooter: int
    shot: np.ndarray  # (2,) shot location (shooter's feet)
    goal_x: float
    features: dict

    @property
    def posts(self) -> np.ndarray:
        mid = WIDTH / 2
        return np.array(
            [[self.goal_x, mid - GOAL_HALF], [self.goal_x, mid + GOAL_HALF]]
        )

    @property
    def opponents(self) -> np.ndarray:
        team = self.teams[self.shooter]
        return np.flatnonzero((self.teams != team) & (self.teams >= 0))

    def in_cone(self) -> np.ndarray:
        """Opponents (goalkeeper excluded) inside the triangle shot location - posts."""
        tri = np.vstack([self.shot, self.posts]).astype(np.float32)
        return np.array(
            [
                i
                for i in self.opponents
                if i != self.keeper
                and cv2.pointPolygonTest(
                    tri, tuple(map(float, self.pitch_xy[i])), False
                )
                >= 0
            ],
            dtype=int,
        )

    def nearest_opponent(self) -> int:
        d = np.hypot(*(self.pitch_xy[self.opponents] - self.shot).T)
        return int(self.opponents[np.argmin(d)])

    def to_image(self, xy: np.ndarray) -> np.ndarray:
        inv = np.linalg.inv(self.homography)
        return cv2.perspectiveTransform(
            np.asarray(xy, np.float32).reshape(1, -1, 2), inv
        )[0]


def load(shot_id: str) -> ShotGeometry:
    frames = pd.read_parquet(XD / "cache/xgd_full4/E3/frames.parquet")
    frame = frames[(frames.shot_id == shot_id) & (frames.offset_ms == 0)].iloc[0]
    features = pd.read_parquet(XD / "cache/xgd_full4_feet_t0/E3/features.parquet")
    feat = features[(features.shot_id == shot_id) & (features.offset_ms == 0)].iloc[0]
    pitch_xy = np.asarray(frame.det_pitch_xy, float).reshape(-1, 2)
    shot = np.array([feat.shot_x, feat.shot_y])
    classes = np.asarray(frame.det_class)
    return ShotGeometry(
        homography=np.asarray(frame.homography, float).reshape(3, 3),
        pitch_xy=pitch_xy,
        boxes=np.asarray(frame.det_xyxy, float).reshape(-1, 4),
        teams=np.asarray(frame.det_team, int),
        keeper=int(np.flatnonzero(classes == "goalkeeper")[0]),
        shooter=int(np.argmin(np.hypot(*(pitch_xy - shot).T))),
        shot=shot,
        goal_x=LENGTH if feat.attack_right else 0.0,
        features={
            "distance": float(feat.dist),
            "angle": float(np.degrees(feat.angle)),
            "def_cone": int(feat.ff_def_cone),
            "near_opp": float(feat.ff_near_opp),
            "opp_3m": int(feat.ff_opp_3m),
            "gk_depth": float(feat.gk_depth),
            "gk_off_centre": float(feat.gk_off_line_abs),
        },
    )


def check(geo: ShotGeometry) -> None:
    """The drawing recomputes the cone and nearest opponent; they must match the cached features."""
    f = geo.features
    assert len(geo.in_cone()) == f["def_cone"], (geo.in_cone(), f)
    near = np.hypot(*(geo.pitch_xy[geo.nearest_opponent()] - geo.shot))
    assert abs(near - f["near_opp"]) < 0.15, (near, f)


# ---------------------------------------------------------------- on the frame


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def _label(
    d: ImageDraw.ImageDraw, xy, text: str, color, alpha: float, size: int = 22
) -> None:
    font = _font(size, bold=True)
    x, y = xy
    box = d.textbbox((x, y), text, font=font)
    pad = 5
    d.rounded_rectangle(
        [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad],
        6,
        fill=(16, 18, 22, int(225 * alpha)),
    )
    d.text((x, y), text, font=font, fill=(*color, int(255 * alpha)))


def _ring(d: ImageDraw.ImageDraw, box, color, alpha: float, width: int = 4) -> None:
    x0, y0, x1, y1 = box
    cx, w = (x0 + x1) / 2, (x1 - x0)
    d.ellipse(
        [cx - w * 0.75, y1 - w * 0.28, cx + w * 0.75, y1 + w * 0.28],
        outline=(*color, int(255 * alpha)),
        width=width,
    )


def freeze_overlay(frame: np.ndarray, geo: ShotGeometry, t: float) -> Image.Image:
    """The frame at contact with the measured geometry; elements appear one by one as t (seconds) grows."""

    def a(start: float) -> float:
        return float(np.clip((t - start) / 0.35, 0, 1))

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    dim = Image.new("RGBA", image.size, (0, 0, 0, int(95 * a(0.0))))
    image.alpha_composite(dim)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = geo.features
    shot_px = geo.to_image(geo.shot[None])[0]
    posts_px = geo.to_image(geo.posts)

    # 1 shooting cone and distance to the goal centre
    cone = [tuple(shot_px), *map(tuple, posts_px)]
    d.polygon(
        cone,
        fill=(*CONE, int(70 * a(0.2))),
        outline=(*CONE, int(230 * a(0.2))),
        width=2,
    )
    centre_px = geo.to_image(np.array([[geo.goal_x, WIDTH / 2]]))[0]
    d.line([tuple(shot_px), tuple(centre_px)], fill=(*CONE, int(200 * a(0.6))), width=2)
    along = shot_px + 0.55 * (centre_px - shot_px)
    _label(
        d,
        (along[0] - 60, posts_px[:, 1].max() + 26),
        f"{f['distance']:.1f} m to goal · {f['angle']:.0f}° goal angle",
        CONE,
        a(0.6),
        20,
    )

    # 2 shooter
    _ring(d, geo.boxes[geo.shooter], SHOOTER, a(0.2))
    sx0, sy0, sx1, sy1 = geo.boxes[geo.shooter]
    _label(
        d,
        (sx0 - 190, sy1 + 18),
        "shooter (feet = shot location)",
        (134, 182, 239),
        a(0.2),
        20,
    )

    # 3 defenders in the cone, nearest opponent and the 3 m circle
    for i in geo.in_cone():
        _ring(d, geo.boxes[i], DEFENDER, a(1.0))
    circle = (
        geo.shot
        + 3.0
        * np.c_[
            np.cos(np.linspace(0, 2 * np.pi, 60)), np.sin(np.linspace(0, 2 * np.pi, 60))
        ]
    )
    d.line(
        [tuple(p) for p in geo.to_image(circle)],
        fill=(*DEFENDER, int(200 * a(1.4))),
        width=2,
    )
    near = geo.nearest_opponent()
    _ring(d, geo.boxes[near], DEFENDER, a(1.4))
    top = min(geo.boxes[i][1] for i in [near, *geo.in_cone()])
    _label(
        d,
        (shot_px[0] - 430, top - 70),
        f"{f['def_cone']} defender in cone · nearest {f['near_opp']:.1f} m · {f['opp_3m']} within 3 m",
        (247, 180, 150),
        a(1.0),
        20,
    )

    # 4 goalkeeper and distance off the line
    _ring(d, geo.boxes[geo.keeper], KEEPER, a(1.8))
    gk = geo.pitch_xy[geo.keeper]
    line_pt = geo.to_image(np.array([[geo.goal_x, gk[1]], gk]))
    d.line([tuple(p) for p in line_pt], fill=(*KEEPER, int(255 * a(1.8))), width=3)
    kx0, ky0, kx1, ky1 = geo.boxes[geo.keeper]
    _label(
        d,
        (kx0 - 90, ky0 - 46),
        f"GK {f['gk_depth']:.1f} m off line",
        (250, 205, 110),
        a(1.8),
        20,
    )

    image.alpha_composite(layer)
    title = ImageDraw.Draw(image)
    _label(
        title,
        (24, 64),
        "Freeze at contact: what the geometry model measures",
        (255, 255, 255),
        a(0.0),
        18,
    )
    return image


# ---------------------------------------------------------------- from above


def diagram(geo: ShotGeometry, path, inset: bool = False) -> None:
    """Top-down view of the attacking third with the measured inputs; inset=True drops the text panel."""
    right = geo.goal_x == LENGTH
    flip = (
        (lambda xy: np.asarray(xy, float))
        if right
        else (lambda xy: np.c_[LENGTH - np.asarray(xy)[..., 0], np.asarray(xy)[..., 1]])
    )
    shot = flip(geo.shot[None])[0]
    pts = flip(geo.pitch_xy)
    posts = flip(geo.posts)
    fig, ax = plt.subplots(figsize=(3.2, 3.1) if not inset else (3.2, 3.2), dpi=150)
    fig.patch.set_alpha(0 if inset else 1)
    ax.set_facecolor("#2f6b3a" if inset else "#f5f4f1")
    line = "#ffffff" if inset else "#85837d"
    x0 = 70 if inset else 76
    for patch in [
        Rectangle((x0, 0), LENGTH - x0, WIDTH, fill=False, ec=line, lw=1),
        Rectangle(
            (LENGTH - 16.5, WIDTH / 2 - 20.16), 16.5, 40.32, fill=False, ec=line, lw=1
        ),
        Rectangle(
            (LENGTH - 5.5, WIDTH / 2 - 9.16), 5.5, 18.32, fill=False, ec=line, lw=1
        ),
        Arc(
            (LENGTH - 11, WIDTH / 2), 18.3, 18.3, theta1=127, theta2=233, ec=line, lw=1
        ),
    ]:
        ax.add_patch(patch)
    ax.plot(
        [LENGTH, LENGTH],
        posts[:, 1],
        color="#0b0b0b" if not inset else "#ffffff",
        lw=4,
        solid_capstyle="butt",
    )
    ax.add_patch(
        Polygon(
            np.vstack([shot, posts]),
            closed=True,
            fc="#ffffff" if inset else "#d9d7d1",
            ec="#52514e",
            lw=0.8,
            alpha=0.55,
        )
    )
    ax.add_patch(Circle(shot, 3.0, fill=False, ec="#eb6834", lw=1, ls=(0, (3, 2))))
    ax.plot(
        [shot[0], LENGTH],
        [shot[1], WIDTH / 2],
        color="#0b0b0b" if not inset else "#ffffff",
        lw=0.9,
        ls=(0, (2, 2)),
    )
    team = geo.teams[geo.shooter]
    cone = set(geo.in_cone().tolist())
    for i, (x, y) in enumerate(pts):
        if not x0 - 1 < x < LENGTH + 1:
            continue
        if i == geo.keeper:
            color, size = "#eda100", 60
        elif i == geo.shooter:
            continue
        elif geo.teams[i] == team:
            color, size = "#86b6ef", 34
        else:
            color, size = "#eb6834", 34
        ax.scatter(
            x,
            y,
            s=size,
            color=color,
            ec="#ffffff" if i not in cone else "#0b0b0b",
            lw=1.6 if i in cone else 0.8,
            zorder=4,
        )
    ax.scatter(*shot, s=80, color="#2a78d6", ec="white", lw=1.2, zorder=5, marker="o")
    gk = pts[geo.keeper]
    ax.plot([gk[0], LENGTH], [gk[1], gk[1]], color="#eda100", lw=1.6)
    f = geo.features
    text = (
        "none" if inset else "#0b0b0b"
    )  # the inset sits next to the same numbers on the frame
    mid = (shot + np.array([LENGTH, WIDTH / 2])) / 2
    ax.text(
        mid[0] - 1.0,
        mid[1] - 2.6,
        f"{f['distance']:.1f} m",
        fontsize=7.5,
        color=text,
        ha="center",
    )
    ax.text(
        shot[0] - 1.5,
        shot[1] + 3.6,
        f"{f['angle']:.0f}°",
        fontsize=7.5,
        color=text,
        ha="right",
    )
    ax.text(
        gk[0] + 1.5,
        gk[1] + 4.2,
        f"GK {f['gk_depth']:.1f} m",
        fontsize=7.5,
        color=text,
        ha="center",
    )
    if not inset:
        from matplotlib.lines import Line2D

        def dot(color, label, edge="white", marker="o", size=7):
            return Line2D(
                [], [], ls="", marker=marker, ms=size, mfc=color, mec=edge, label=label
            )

        handles = [
            dot("#2a78d6", "shooter (shot location)", size=8),
            dot("#86b6ef", "teammate"),
            dot("#eb6834", "opponent"),
            dot("#eb6834", "opponent in shooting cone", edge="#0b0b0b"),
            dot("#eda100", "goalkeeper (depth off line)"),
            Line2D(
                [], [], color="#eb6834", ls=(0, (3, 2)), label="3 m pressure radius"
            ),
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            fontsize=6.3,
            frameon=False,
            ncol=2,
            handletextpad=0.3,
            columnspacing=0.8,
        )
    ax.set_xlim(x0 - 0.5, LENGTH + 1.5)
    span = 22 if inset else 14
    ax.set_ylim(
        WIDTH / 2 + span, WIDTH / 2 - span
    )  # far touchline at the top, as on screen
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, transparent=inset)
    plt.close(fig)
