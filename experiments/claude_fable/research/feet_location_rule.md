# Shot location from the shooter's feet: the frozen rule `feet_t0` (2026-10-22)

Frozen and committed before any further label read. Parameters, hashes and evidence are in
`research/feet_location_rule.json`. The code is `xgpoc/detector/location.py` (`FEET_T0`) and
`xgpoc/stack_location/inputs.py`, committed in f21a780. The cache is built by `scripts/feet_e3.py` and checked by
`scripts/location_check.py`.

## Status

This is a definition, not a result. The variant came out of the owner-ordered error decomposition of the dd2c sweep
(2026-10-22, decision-log row `err-decomp`). It was chosen after reading development labels (328 training matches)
and the labels of the 172 held-out matches. Those 172 matches were already spent for detector-input variants: by the
A1 test, the xg-stack test and the 2026-10-18 sweep. **Every number obtained on them for this variant is descriptive.
A confirmatory claim about `feet_t0` needs a fresh test set: matches that no model, rule or variant choice has read.**

## Definition

The rule applies to every E3 feature row: t0, and -400 ms. It runs on the frames of the retrained detector's E3 cache
(`cache/xgd_full4/E3`, dd2c YOLO11l at 1280, unchanged).

1. **Ball order.** The ball comes from the first frame with a ball on the pitch, in the order 0, -400, -800 ms. No frame
   after the row's own frame is used (`SourceConfig("feet_t0", (0, -400, -800)).order`).
   - The t0 row uses 0, -400, -800 ms: the t0 frame's own ball first. E3 used -400, -800, 0.
   - The -400 ms row uses -400, -800 ms, as E3 does.

   The ball is placed by its own frame's homography. Its image box keeps that frame's pixel coordinates
   (`extract3.with_ball`).
2. **Shooter.** When a ball exists and a player (class "player") has a pitch position, the shooter is the player whose
   image box is nearest the image centre of that ball's box. The distance is point to rectangle, in pixels, and is 0
   inside the box. On a tie, the first player in detection order wins (`location.image_shooter`). Without a ball, or
   without a player, the shooter is xg-detector's: the player nearest either goal centre (`shooter_by_goal`).
3. **Shot location.** With a ball, the shot location is the shooter's projected position, his feet. It is never the
   ball's projection, which lies metres away for a ball in the air. Without a ball the shot location is again the
   shooter's feet, as in xg-detector.
4. **Attacked goal and goalkeeper.** Both are chosen from that shot location exactly as `features.situation` chooses
   them: the goal nearer the shot location; the detected goalkeeper nearest the attacked goal in the attacked half,
   else the proxy.
5. **Raw features.** The `features.COLUMNS` come from that situation (`location.situation_features`).
   - `ball_ok`, `ball_fallback` and `ball_source` describe the ball the order picked.
   - `shot_from_ball` = 1 whenever a ball exists, so it means "the shooter was picked by a ball".
   - `shooter_ball_dist` is the pitch distance from the shooter's feet to the ball's projection.
6. **Stack table: t0 ball flags kept from E3.** In the xg-stack table (`stack_location.inputs.e3_table`), the t0
   ball-source columns are E3's: `ball_none@0`, `ball_own@0`, `ball_m400@0`, `ball_m800@0` and `shot_from_ball@0`.
   They are the sanitised `ball_source` and `shot_from_ball` of the E3-filled t0 frame (order -400, -800, 0) under
   `features.situation`. Every other column, at both offsets, comes from rules 1-5 and the sanitation run on the feet
   situation.
   - **Decision: the frozen variant keeps dd2c's t0 ball flags.** On development data the gain of the feet variant is
     the location's, not the flags'. With dd2c's t0 flags kept, stack_ALL - R moves from -0.0056 to -0.0116 (C6
     read-out, 328 matches). The same arm, feet_t0 with dd2c flags minus dd2c, gives ALL -0.0061 [-0.0090; -0.0029]
     (`results/xgs_stack_feet_t0_flags`, decision-log row `err-decomp`). The held-out sweep
     (`results/sweep_feet_2026_10`) used this handling.
   - The feet location is never masked, so `shot_masked@0` has zero variance. A C6 design drops it.

**Variant name:** `feet_t0` (`xgpoc.detector.location.FEET_T0`, `RULES["feet_t0"]`).

## Cache

`cache/xgd_full4_feet_t0/E3` (`scripts/feet_e3.py`, at commit f21a780, clean tree):

| File | sha256 |
|---|---|
| `features.parquet` (22,060 rows, 11,030 shots, t0 and -400 ms) | `93b4b580dcec76ee2c315440e9b7d069bb5e62f07dc3ebb06db69b328b219a1d` |
| `frames.parquet` (link to `cache/xgd_full4/E3/frames.parquet`) | `913b1bd9bd3a737a28126308113c44f060d2479a5351ac82bd8cccb666c19e78` |
| `complete.json` | `6cbe28896429ab630af9f248b63820a1e498645502029f49df110f4a2593146e` |

`features.parquet` is byte-identical to the development cache that the 2026-10-22 monkey-patch built
(`cache/xgd_full5_feet_t0/E3`, `scripts/feet_patch.py` with `apply(t0_ball=True)`, superseded by the library rule). The stack tables under
this rule equal the patched tables exactly:
- the training population, 7,062 shots, E3 columns sha256 `db05127b…`;
- the held-out table, 3,819 shots, `f3c22b93…`.

Those tables are behind `results/xgs_stack_feet_t0_flags` and `results/sweep_feet_2026_10`.

## What the frozen code guarantees

No frozen module was edited, so every pinned code hash and every E3 replay stays as it was:
- `features.py`, `extract3.py` and `stack/inputs.py`;
- `stack_heldout/testset.py`;
- the xg-stack protocol, the xg-stack test hash list and the detector re-run records.

The new code copies three frozen pieces with the rule as an argument: `shot_features`, `filled_frame` / `replay`, and
`e3_table` / `build_population` / `heldout_table`. Under the `E3` rule the copies reproduce the frozen functions on every
shot of `cache/xgd_full3/E3` and `cache/xgd_full4/E3`: feature rows, E3 table and population. `scripts/location_check.py`
checks this, along with the cache and table equalities above.

## Development evidence (descriptive; it chose the variant)

All of the following come from decision-log row `err-decomp` (commit 99d48d2):
- Location error against StatsBomb drops from 2.62 m to 1.16 m. The share at 5 m or more drops from 32% to 15%, and
  headers drop from 13.0 m to 2.4 m.
- 172 held-out matches (spent): stack_Q - G-combo is -0.0110 [-0.0153; -0.0065], against -0.0069 for dd2c's best.

None of these numbers is a test of this rule.
