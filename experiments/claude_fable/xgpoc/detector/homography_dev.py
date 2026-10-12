"""Development loop of the homography fix (xg-detector 08): training-match shot frames, homography variants, scores.

Frames: t0 frames of shots in the xg-vendor split plan's **training** matches only (role "training", 328 matches); the
gate and evaluation matches are never sampled. They are split by the plan's home-team fold: folds 0-2 are `dev`, where
variants are compared and chosen, and folds 3-4 are `confirm`, where the chosen ones are checked once more.

A variant is the package's keypoint input (letterbox or stretch), the keypoint confidence the homography uses, and the
line refinement on or off, all run through
`tactifoot_vision.pitch.HomographyEstimator.fit` on the cached keypoints of its input. `C4` is the extraction before
this ticket: the homography cached by ticket 02's full C4 run (letterbox, keypoint confidence 0.5, no refinement).

Scores per frame (`xgpoc.detector.lines`): the line agreement of the projected markings, and two checks that use no
lines, players on the pitch and the players' height ratio. Per variant (`summarise`):
- `homography`: share of frames with a homography;
- `good`: share of **all** frames whose homography has line agreement >= `GOOD` (a method cannot look good by refusing
  hard frames); `unconfirmed` = `homography` - `good`;
- `agreement_median` among scorable frames, `height_log_median` (median |log height ratio| over frames with a
  homography and players; lower is a more plausible scale), `players_on_pitch` (mean share).
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import supervision as sv

DEV_FOLDS = (0, 1, 2)
CONFIRM_FOLDS = (3, 4)
GOOD = 0.5                      # line agreement of a usable homography: a 0.5 m shift already drops a synthetic frame to 0.51
INSTANCE_CONF = 0.1             # the keypoint instance threshold of C1-C4 (ticket 01)


@dataclass(frozen=True)
class Variant:
    name: str
    keypoint_input: str                     # "letterbox" | "stretch"
    min_confidence: float = 0.5             # keypoints the homography uses
    refine_with_lines: bool = False

    def estimator(self):
        from tactifoot_vision.pitch import HomographyEstimator

        return HomographyEstimator(min_confidence=self.min_confidence, ransac_threshold=10.0,
                                   refine_with_lines=self.refine_with_lines)

    def settings(self) -> dict:
        return asdict(self)


def sample_training_frames(plan: pd.DataFrame, index: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """`n` random shots (seeded) of the plan's training matches: shot_id, game, fold, subset (dev / confirm), frame_path.

    `index` needs shot_id, game and frame_path only; `plan` game, role and fold."""
    training = plan.loc[plan.role == "training", ["game", "fold"]]
    if training.game.duplicated().any():
        raise ValueError("a training game appears twice in the split plan")
    shots = index[["shot_id", "game", "frame_path"]].merge(training, on="game", how="inner", validate="many_to_one")
    shots = shots.sort_values("shot_id").sample(n, random_state=seed).reset_index(drop=True)
    if not shots.fold.isin(DEV_FOLDS + CONFIRM_FOLDS).all():
        raise ValueError(f"unexpected folds {sorted(set(shots.fold) - set(DEV_FOLDS + CONFIRM_FOLDS))}")
    shots["subset"] = np.where(shots.fold.isin(DEV_FOLDS), "dev", "confirm")
    return shots


def keypoint_row(keypoints: sv.KeyPoints) -> dict:
    """The first (most confident) instance of a keypoint prediction as a cache row: xy, confidences, instance confidence."""
    if len(keypoints) == 0:
        return {"kp_xy": None, "kp_conf": None, "kp_instance_conf": np.nan}
    return {"kp_xy": [float(v) for v in keypoints.xy[0].ravel()], "kp_conf": [float(v) for v in keypoints.confidence[0]],
            "kp_instance_conf": float(keypoints.data["box_confidence"][0])}


def keypoints_from_row(row) -> sv.KeyPoints | None:
    """The cached first instance as `sv.KeyPoints`, or None (no instance, or instance confidence < `INSTANCE_CONF`)."""
    xy = row["kp_xy"]
    if xy is None or (isinstance(xy, float) and np.isnan(xy)) or not row["kp_instance_conf"] >= INSTANCE_CONF:
        return None
    return sv.KeyPoints(xy=np.asarray(xy, np.float32).reshape(1, -1, 2),
                        confidence=np.asarray(row["kp_conf"], np.float32).reshape(1, -1))


def summarise(per_frame: pd.DataFrame) -> pd.DataFrame:
    """One row per variant and subset (dev, confirm, all) with the scores of the module docstring."""
    rows = []
    for subset, frames in [("all", per_frame), *per_frame.groupby("subset")]:
        for variant, group in frames.groupby("variant", sort=False):
            has = group.has_homography.astype(bool)
            good = (group.agreement >= GOOD) & has
            log_height = np.abs(np.log(group.height_ratio[has]))
            rows.append({"variant": variant, "subset": subset, "frames": len(group),
                         "homography": has.mean(), "scorable": group.agreement.notna().mean(),
                         "good": good.mean(), "unconfirmed": (has & ~good).mean(),
                         "agreement_median": group.agreement.median(), "agreement_mean": group.agreement.mean(),
                         "recall_median": group.recall.median(),
                         "height_log_median": float(np.nanmedian(log_height)) if log_height.notna().any() else np.nan,
                         "players_on_pitch": group.players_on_pitch[has].mean()})
    return pd.DataFrame(rows)
