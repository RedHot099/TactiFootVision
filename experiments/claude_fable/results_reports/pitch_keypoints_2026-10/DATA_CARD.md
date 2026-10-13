# Data card: SoccerNet pitch keypoints (pitch-keypoints 01)

Built 2026-10-13T10:06:03+00:00 by `scripts/pk_build_dataset.py` at git ffa3438; dataset `/mnt/storage/tactifoot_research/pitch_keypoints_sn`; labels digest `d5c2af0e572c40c9`.

## What it is

One instance per image (class `pitch`): the box of the pitch area in the frame and the 32 vertices of `tv.pitch.SoccerPitch()` (corner origin, x along the length, y = 0 at the far touchline; the Roboflow layout of `data/keypoints`) as (x, y, v), v = 2 in the frame and in front of the camera, else 0 with x = y = 0. Labels are projections through a homography fitted to SoccerNet's annotated pitch lines (`xgpoc/pitchkp/fit.py`); nothing is hand-labelled. `kpt_shape [32, 3]`, `flip_idx` as in `data/keypoints/data.yaml`. Images stored at 960 x 540 (GSR frames downscaled from 1920 x 1080; Calibration images are 960 x 540 originals).

## Sources and licences

- SoccerNet Game State Reconstruction 2024 (`gamestate-2024`): 30 s clips of 12 games of the 2019 Swiss Super League (SoccerNet-Tracking), main camera; every 5th frame (1, 6, 11, ...) kept. Per-frame pitch lines and persons' `bbox_pitch`.
- SoccerNet Camera Calibration 2023 (`calibration-2023`): broadcast frames of SoccerNet-v2 games (EPL, La Liga, Ligue 1, Bundesliga, Serie A, UCL; 2014-2017), live and replay views, per-image pitch lines.
- Both via the `SoccerNet` package downloader (public data, no NDA password). SoccerNet data are for research use under the SoccerNet terms (non-commercial; broadcast rights stay with their owners); the derived labels inherit those terms.

## Rejection rule (720p pixels)

An image is kept when its line fit exists (>= 9 annotated ground-plane points incl. a straight line, a camera view: no mirror image), RMS residual <= 4.0 px, largest residual <= 16.0 px, every visible vertex's 2-D standard deviation <= 10.0 px (residual covariance, floor 1.0 px; underdetermined fits are infinite), the pitch area is in the frame, and the view is from the main side: up the frame at its bottom centre points to the far touchline within 66 degrees (pitch direction y component < -0.4). SoccerNet names lines in pitch coordinates, so reverse-angle and along-the-pitch views would carry labels rotated by 180 or 90 degrees against main-camera views of look-alike pictures.

## Counts

| source/split | images | groups | label rule kept | rejected (first rule broken) | - xG held-out games | - GSR evaluation games | - duplicates | exported |
|---|---|---|---|---|---|---|---|---|
| calib/test | 3141 | 54 | 2005 | vertex sd 430, no fit 369, view: along the pitch 178, rms 128, view: opposite side 27, mirror-inconsistent 3, max residual 1 | 2005 | 2005 | 2005 | 2005 |
| calib/train | 16463 | 290 | 10537 | vertex sd 2072, no fit 1787, view: along the pitch 1039, rms 828, view: opposite side 187, mirror-inconsistent 11, max residual 2 | 7585 | 7585 | 7585 | 7585 |
| calib/valid | 3212 | 55 | 2063 | vertex sd 454, no fit 349, view: along the pitch 182, rms 130, view: opposite side 28, mirror-inconsistent 5, max residual 1 | 1658 | 1658 | 1658 | 1658 |
| gsr/test | 7350 | 49 | 6653 | no fit 448, rms 195, vertex sd 38, mirror-inconsistent 14, max residual 2 | 6653 | 6653 | 6653 | 6653 |
| gsr/train | 8550 | 57 | 8082 | rms 241, no fit 146, vertex sd 77, mirror-inconsistent 2, max residual 2 | 8082 | 8082 | 8082 | 8082 |
| gsr/valid | 8700 | 58 | 8243 | rms 194, vertex sd 130, no fit 119, max residual 8, mirror-inconsistent 6 | 8243 | 8243 | 8243 | 8243 |

Groups are GSR clips and Calibration games; a group never spans two splits. The exclusion columns give the images left after each step (train and val only; test splits are not filtered).

| export split | images | by source |
|---|---|---|
| test_calib | 2005 | calib 2005 |
| test_gsr | 6653 | gsr 6653 |
| train | 15667 | gsr 8082, calib 7585 |
| val | 9901 | gsr 8243, calib 1658 |

`train` and `val` are in `data.yaml`; `test_gsr` and `test_calib` are for evaluation only.

## Residuals of the kept images (720p px)

| source/split | RMS p50 | RMS p95 | vertex sd p50 | vertex sd p95 |
|---|---|---|---|---|
| calib/test | 1.49 | 2.51 | 3.93 | 8.22 |
| calib/train | 1.53 | 2.63 | 4.22 | 8.08 |
| calib/valid | 1.60 | 2.64 | 4.42 | 8.51 |
| gsr/test | 0.97 | 1.75 | 2.95 | 6.21 |
| gsr/train | 0.77 | 2.30 | 2.95 | 4.53 |
| gsr/valid | 0.97 | 2.37 | 2.52 | 5.86 |

## Checks

GSR's labels carry no camera parameters (only `object` and `pitch` annotations), so the spec's check of camera-projected against line-fit vertices is replaced by the position check below.

GSR positions: each kept frame's persons (`bbox_image` bottom-middle through the line fit vs `bbox_pitch` bottom-middle, centre origin + (52.5, 34)); per-frame median error in metres:

| frames | n | p50 | p90 | p99 | < 0.5 m | < 1 m | < 2 m |
|---|---|---|---|---|---|---|---|
| all fitted/test | 6817 | 0.35 | 0.90 | 4.75 | 0.695 | 0.902 | 0.952 |
| all fitted/train | 8376 | 0.36 | 1.12 | 4.04 | 0.662 | 0.879 | 0.958 |
| all fitted/valid | 8466 | 0.38 | 1.06 | 4.50 | 0.640 | 0.877 | 0.945 |
| kept/test | 6569 | 0.34 | 0.78 | 2.62 | 0.717 | 0.928 | 0.971 |
| kept/train | 8082 | 0.35 | 0.97 | 2.84 | 0.685 | 0.906 | 0.981 |
| kept/valid | 8131 | 0.37 | 0.96 | 3.64 | 0.656 | 0.895 | 0.957 |
| rejected/test | 248 | 2.78 | 9.70 | 118.46 | 0.120 | 0.205 | 0.442 |
| rejected/train | 294 | 2.13 | 13.39 | 36.81 | 0.081 | 0.205 | 0.398 |
| rejected/valid | 335 | 1.09 | 7.80 | 66.19 | 0.251 | 0.456 | 0.654 |

Axis directions (GSR valid): homography from persons' feet pixels to (s_x x + 52.5, s_y y + 34), then the line-fit vertices mapped through it vs the package vertices, median of per-frame medians: s = +1+1: 0.40 m; s = +1-1: 18.38 m; s = -1+1: 82.24 m; s = -1-1: 82.89 m.

Horizontal flip (`flip_idx`, rule 7): all 44198 fitted images refitted from mirrored annotations; same visible set on 43610; kept images' largest vertex difference 0.570527 px (median 0.00000331 px).

Mirror images: none can be kept (the fit only accepts det(H) w > 0 at every annotated point).

## Visible vertices per kept image

| source/split | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| calib/test | 1 | 1 | 42 | 52 | 118 | 115 | 98 | 187 | 296 | 268 | 406 | 273 | 127 | 14 | 4 | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| calib/train | 4 | 21 | 98 | 191 | 383 | 448 | 331 | 603 | 1046 | 1136 | 1672 | 1080 | 478 | 62 | 21 | 2 | 2 | 4 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| calib/valid | 5 | 4 | 31 | 39 | 85 | 115 | 87 | 144 | 257 | 241 | 354 | 186 | 101 | 7 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gsr/test | 0 | 1 | 163 | 211 | 725 | 910 | 353 | 358 | 1068 | 887 | 652 | 878 | 372 | 70 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gsr/train | 0 | 0 | 112 | 33 | 772 | 1143 | 289 | 376 | 1459 | 803 | 871 | 1194 | 567 | 109 | 111 | 118 | 95 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| gsr/valid | 0 | 1 | 68 | 57 | 480 | 1149 | 367 | 459 | 1126 | 1011 | 894 | 1094 | 1086 | 221 | 105 | 85 | 22 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Share of kept images where each vertex is visible: 0: 0.33, 1: 0.34, 2: 0.32, 3: 0.19, 4: 0.06, 5: 0.01, 6: 0.34, 7: 0.26, 8: 0.33, 9: 0.30, 10: 0.32, 11: 0.32, 12: 0.23, 13: 0.28, 14: 0.29, 15: 0.28, 16: 0.05, 17: 0.34, 18: 0.36, 19: 0.35, 20: 0.23, 21: 0.35, 22: 0.36, 23: 0.26, 24: 0.37, 25: 0.37, 26: 0.33, 27: 0.19, 28: 0.06, 29: 0.01, 30: 0.27, 31: 0.28.

## Game identity and the xg-vendor split plan

Council C3 (Q2): train and val hold no frame of the 172 xG held-out matches (xg-vendor split plan, roles `evaluation` and `gate`) where the game is known, no game of GSR valid or test, and no duplicate of a frame in a higher-priority split (test > val > train).

- Calibration test: games by plan role {'training': 21, 'evaluation': 18, 'gate': 15}; images passing the label rule by role {'training': 778, 'evaluation': 701, 'gate': 526}.
- Calibration train: games by plan role {'training': 214, 'evaluation': 76}; images passing the label rule by role {'training': 7585, 'evaluation': 2952}.
- Calibration valid: games by plan role {'training': 46, 'evaluation': 9}; images passing the label rule by role {'training': 1658, 'evaluation': 405}.
- GSR game ids (an index without match names): train ['4', '6', '9'], val ['2', '3', '5'], test ['11', '7', '8']. GSR vs the xG plan and vs Calibration: not matchable by name (GSR game_id is an index without a match name); no overlap by construction: GSR is the 2019 Swiss Super League (SoccerNet-Tracking), Calibration-2023 and the xG plan are SoccerNet-v2 games (2014-2017 EPL, La Liga, Ligue 1, Bundesliga, Serie A, UCL), so step 2 removes no Calibration image.
- Duplicates, among the images passing the label rule: 55545 pairs within 4 of 64 difference-hash bits, mostly neighbouring frames of one clip or game; 7 of them across splits or sources (0 across sources), of which 0 are the same picture (thumbnail correlation >= 0.95; 0 across sources).

The keypoint model sees no goal label and no StatsBomb data; the held-out matches stay in `test_calib` (evaluation only).

