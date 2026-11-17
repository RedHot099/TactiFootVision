# The pitch homography: evidence for the line-network estimator

This document is the evidence for ADR 0005 and for the homography numbers in the README. The full records stay on the local research branch `research/homography-final` at commit `1492a5af7641` (not pushed): its `experiments/homography_bench` (the benchmark, its protocol `METRICS.md`, every scored record `experiments/homography_bench/results/<row>/<split>/metrics.json` and the comparisons in `experiments/homography_bench/results/compare/`) and `experiments/homography_joint` (the hill climb, its report `REPORT.md`, its rules and its own records in `experiments/homography_joint/results/`). Each section below names the record a number comes from. Paths are relative to that commit.

## 1. The question

Can the package's pitch homography improve on the keypoint estimator it shipped with (B0: a pitch keypoint network, RANSAC to 32 landmarks, a chamfer refinement on a white-line mask, Savitzky-Golay smoothing)? Every frame of a broadcast clip should have a matrix that puts the players' feet and the pitch markings where they are, steadily, at no more cost. The question was answered with a protocol fixed before any candidate was scored.

## 2. The protocol in short

`experiments/homography_bench/METRICS.md`.

- **Data:**
  - training: GSR train and SN-Calib train only;
  - development and every choice: GSR valid (58 clips, every frame, plus 20 cut pairs) and SN-Calib valid;
  - held out: GSR test (49 clips) and SN-Calib test, scored for B0 first and for the final candidate once.
- **Headline numbers** (on the `final` track) and their minimum effects:
  1. AUC5: the area under the success curve of the feet's pitch error up to 5 m, per clip, then the mean over clips; minimum effect +0.02.
  2. p90 of that error, with a failed frame counted as 10 m; minimum effect 10 % of B0's.
  3. Line accuracy@5 px: the mean of GSR and SN-Calib; minimum effect +2 points.
  4. Jitter, D1b p95: the truth-free second difference over three frames; minimum effect −20 %.
  5. GPU ms per frame.
- **Guardrails, all against B0:**
  - availability, valid share, jitter and speed coverage;
  - gross errors, beyond-horizon, motion lag, region errors, visible area, the worst SN-Calib view;
  - speed error, implausible speeds, cut recovery;
  - CPU and GPU cost, VRAM;
  - the causal track;
  - the AUC5 spread over three seeds. For this estimator the guardrail is met trivially, since it is deterministic on the same weights (spread 0.0); it says nothing about the training run's variance, which section 7 measures.
- **The rule.** A candidate beats B0 when every guardrail holds and either
  - (a) a quality number improves (by its minimum effect, with the 95 % paired interval excluding 0), none is worse, and cost is at most 1.25 × B0's; or
  - (b) cost improves and no quality number is worse.
- **Cost protocols:**
  - `gpu-v2`: networks on the GPU, at most 4 CPU threads;
  - `cpu-v2`: GPU hidden, at most 4 CPU threads;
  - both use 500 frames of three frozen valid clips after 50 warm-up frames, 5 runs.
- **Fixed before the runs it governs** (all committed first on the research branch):
  - the protocol and B0's scores, before any candidate;
  - `KF_RULE.md`, the filter's setting, before the 58-clip runs;
  - `LEAKAGE_RULE.md`, before the leakage check of the held-out sets (nothing flagged);
  - `FREEZE.md`, the configuration and rows of the single held-out run, before that run;
  - `VIEW_RULE.md`, the opposite-side flag and its bar, before the head was trained;
  - `GUARD_RULE.md`, parts 1 to 3 and the amendment, each before its runs.

## 3. The candidates and why this one was kept

The research branch's report (`REPORT.md`; earlier rounds in `REPORT_round1-3.md`) records each step.

- **B0:** the keypoint estimator, the baseline.
- **Line refinement on keypoints**, then **a line network** (`lines`: ResNet-18 + FPN to stride 2, 640 × 360, 20 marking heatmaps) with **one joint fit** of a homography to all its points (E, F, G). G is the line fit alone: Levenberg-Marquardt, soft-L1, closed-form Jacobian, three camera templates only without a previous matrix.
- **The temporal layer.** Three were tried:
  - a pan-tilt-zoom camera model (ptz-J4): rejected, about 8 ms and 2,500 lines for 22 % less causal jitter, 4 points less causal line accuracy;
  - optical flow (flowJ5): no better;
  - a constant-velocity Kalman filter on four image points (K), with its RTS pass as `final`: kept, because it is the steadiest `final` (D1b 0.032 m against Savitzky-Golay's 0.068 m), never smooths across a cut, and costs well under a millisecond.
- **The cut restart.** A frame refused with the previous matrix is solved on its own and taken at ≤ 1 m uncertainty.
- **GK = G + restart + K** was declared, and ported into the package unchanged: `P-lines` equals GK on every frame, at most 7e-13 m apart.
- **The frozen primary row `P-final`:** the package code at `2fd6084`, with weights trained through `tv.train` with the declared recipe (20 epochs, batch 12, lr 6e-4, seed 0).

## 4. Development numbers (GSR valid with its cut pairs, SN-Calib valid)

ADR 0005, "Numbers"; records in `experiments/homography_bench/results/<row>/{valid,calib-valid}/metrics.json`. the research branch's `compare.py B0 P-final` gives `beats_baseline = True` (the seeds guardrail met trivially, as in section 2) on valid (`experiments/homography_bench/results/compare/P-final_vs_B0.md`).

| Row | track | AUC5 | p90 (m) | GSR acc@5 | D1b p95 (m) | SN acc@5 | SN worst @10 |
|---|---|---|---|---|---|---|---|
| B0 | final | 0.819 | 2.428 | 0.661 | 0.264 | 0.285 | 0.046 |
| G | causal | 0.884 | 1.143 | 0.870 | 0.283 | 0.469 | 0.222 |
| GK | final | 0.885 | 1.140 | 0.870 | 0.032 | 0.469 | 0.222 |
| **P-final** | causal | 0.885 | 1.142 | 0.865 | 0.139 | 0.472 | 0.228 |
| **P-final** | final | 0.886 | 1.139 | 0.867 | 0.033 | 0.472 | 0.228 |
| P-final-gweights | final | 0.885 | 1.141 | 0.870 | 0.033 | 0.469 | 0.222 |
| P-final-nofilter | causal | 0.885 | 1.143 | 0.867 | 0.282 | 0.472 | 0.228 |
| P-final-t1 | final | 0.884 | 1.162 | 0.867 | 0.033 | 0.461 | 0.201 |

- **These are development numbers.** Three templates, the 1 m restart threshold, accel 0.15 (the edge of its grid), the hold policy and `max_uncertainty_m` were chosen on them.
- **The first second after a cut** (20 pairs, AUC5): P-final's design GK gives 0.867 causal and 0.868 `final`, against G + Savitzky-Golay's 0.740, which smoothed across the cut.

## 5. The held-out run

`experiments/homography_joint/HELDOUT.md`; freeze `a6a0f1d`, code `2fd6084`. One session, nothing rerun. `compare.py B0 P-final --held-out` (`experiments/homography_bench/results/compare/P-final_vs_B0_heldout.{md,json}`) gives **`beats_baseline = True`**: rules a and b hold, and every guardrail holds. The seeds guardrail, carried over from valid, is met trivially (section 2).

| `final`, GSR test + SN-Calib test | B0 | P-final | difference [95 %] | minimum effect |
|---|---|---|---|---|
| AUC5 | 0.830 | **0.898** | +0.068 [+0.039, +0.107] | 0.020 |
| p90 (m) | 2.921 | **1.102** | −1.820 [−2.891, −0.896] | 0.243 |
| line accuracy@5 (mean of GSR and SN-Calib) | 0.485 | **0.642** | +0.157 [+0.138, +0.182] | 0.020 |
| jitter D1b p95 (m), on the triples shared with B0 | 0.185 | **0.0385** | −0.147 [−0.285, −0.069] | 0.037 |

| Guardrail | B0 | P-final | limit |
|---|---|---|---|
| availability C1 / valid share C1c | 0.995 / 0.950 | 0.999 / 0.999 | ≥ B0 − 1 point |
| gross errors C2 | 0.70 % | 0.33 % | ≤ B0 + 0.3 points |
| beyond horizon C3 | 0.63 % | 0.00 % | ≤ 0.5 % |
| motion lag D3 p90 (m) | 2.89 | 1.34 | ≤ 1.1 × B0 |
| penalty / far side A6 p90 (m) | 1.41 / 3.36 | 0.99 / 1.21 | ≤ 1.1 × B0 |
| speed MAE F1 / implausible speed F3 | 1.39 / 1.26 % | 1.05 / 0.76 % | ≤ 1.05 × B0 / ≤ B0 + 0.1 point |
| visible area A9 IoU | 0.909 | 0.966 | ≥ B0 − 2 points |
| SN-Calib worst view @10 | 0.052 | 0.215 | ≥ B0 − 2 points |
| causal AUC5 / p90 / D3 | 0.830 / 2.92 / 2.93 | 0.897 / 1.11 / 1.35 | AUC5 ≥ B0 − 0.01, p90 ≤ 1.05 × B0, D3 ≤ 1.1 × B0 |
| causal jitter D1b p95 (m), all frames (not a guardrail) | 0.473 | 0.175 | |

- **The shares the README quotes:** `A1_success@0.5` = 0.696 and `A1_success@1` = 0.934, the means over the 49 GSR test clips (`experiments/homography_bench/results/P-final/test/metrics.json`, `final`).
- **SN-Calib availability:** 0.771 on test, against B0's 0.839. Undetermined fits are refused by design.
- **Valid against test:**
  - AUC5 0.886 → 0.898;
  - jitter (D1b p95 on all frames, `metrics.json`) 0.033 → 0.0381 m, optimistic by about 15 % (the comparison's 0.0385 is on the triples shared with B0);
  - GSR line accuracy 0.867 → 0.815, while B0's rose 1 point.

## 6. Cost

Protocols `gpu-v2` and `cpu-v2`. Records: `experiments/homography_bench/results/P-final/cost_{gpu,cpu}{,_images}.json`, window r4d, and B0's records.

| | P-final, video | P-final, single images | B0, video |
|---|---|---|---|
| GPU, ms per frame p50 (p95) | 6.16 (8.9) | 36.3 (58) | 84.6 |
| CPU, ms per frame p50 (p95) | 119 (129) | 129 (150) | 407 |
| VRAM (GB) | 0.23 | 0.24 | 0.42 |

- **An idle-machine repeat** (window r4e): 118.1 ms on the CPU (p95 127.2).
- **The earlier P-lines record of 106 ms** (`a617c4f`) was one fast window: the same weights took 114 ms in another window. The difference lies entirely in the line network's time; the solver takes 4.0 to 4.6 ms in every record (REPORT 13.4, `experiments/homography_joint/results/cost_cpu_ab.json`).

## 7. Training-run variance, and a guard that was tried and removed

REPORT sections 13 to 19; `GUARD_RULE.md`.

- **The recipe trained with five seeds** (seeds 1 to 4 are development rows on valid; `experiments/homography_joint/results/variance_table_five.json`):

  | weights | causal D1b p95 (m) | worst clip, causal D1b p95 (m) | final D1b p95 (m) | AUC5 final |
  |---|---|---|---|---|
  | seed 0 (P-final) | 0.139 | 0.277 (SNGS-084) | 0.033 | 0.886 |
  | seed 1 | 0.178 | 2.063 (SNGS-087) | 0.044 | 0.885 |
  | seed 2 | 0.138 | 0.263 (SNGS-035) | 0.032 | 0.885 |
  | seed 3 | 0.141 | 0.266 (SNGS-035) | 0.032 | 0.885 |
  | seed 4 | 0.613 | 27.580 (SNGS-029) | 0.033 | 0.885 |
  | G's weights | 0.141 | 0.451 (SNGS-087) | 0.033 | 0.885 |

  - **Position error and line accuracy varied little over the five measured trainings** (seeds 0 to 4, `final`, `experiments/homography_joint/results/variance_table_five.json`): AUC5 0.8848 to 0.8859, p90 1.1376 to 1.1410 m (3.4 mm), GSR line accuracy@5 0.8675 to 0.8692, SN-Calib line accuracy@5 0.4656 to 0.4745.
  - **Causal jitter does:** over seeds 0 to 4 it spreads by 0.474 m, and the worst clip by 27.3 m. Over seeds 0 to 2 the spreads were 0.039 m and 1.80 m.
  - **The cause** (`variance_diagnosis.py`): on some clips a network gives points that support a wrong solution in single frames; the filter rejects that fit but returns it, and the RTS pass repairs it (`final` spread 0.012 m).
- **The guard.** On a rejected frame before a reset, the frame's own points chose between the filter's prediction and the fit.
  - **Part 1** (majority support for the prediction) hid real cuts. On valid it handed the prediction to 20 frames of real changes.
  - **Part 2** (`experiments/homography_joint/results/guard_evidence.json`, every rejected frame on valid and TS-WorldCup, four sets of weights): a wrong single fit explains its own frame worse than the prediction (62 of 64), and the first frames of a real change better (177 of 178, one tie). The rule became "the prediction only when it explains more of the frame's points than the fit"; the owner's amendment sent ties to the fit.
  - **The strict rule on valid:** seed spread of causal D1b 0.039 → 0.009 m, worst clip 1.80 → 0.31 m, `final` identical. It changed 15 distinct frames for seed 0: 11 replaced a wrong matrix (one of them within noise: SNGS-026 frame 688, 1.25 → 1.20 m), 1 hid a better one, 3 had no truth (`experiments/homography_joint/results/guard_changes.md`).
  - **Part 3, fresh cuts** (`experiments/homography_joint/results/ts_worldcup/cuts.md`): 85 real cuts joined from TS-WorldCup sequences by a fixed recipe, the four sets of weights with and without the guard.
    - In the first second after the cuts the guard decided 166 frames per set of weights and returned the prediction on none: AUC5 and p90 are identical, and nothing was hidden.
    - Away from the cuts, for seed 0, D1 against the truth rose by 0.012 m, 95 % interval [0.003, 0.024], which fails the rule.
    - That is one frame of one sequence, which the recipe puts into four cut clips, counted as independent by the bootstrap.
  - **The rule said a failure removes the guard: it was removed.** The package returns a rejected fit as it is, as the held-out run measured. The four frozen rows rerun on valid and calib-valid equal their records on both tracks (`experiments/homography_joint/results/compare_runs_*_noguard_ebd8de5.json`).
  - **Part 4, the last attempt** (written before anything was fetched, trained or run). The plan was to judge the guard on untouched footage, the GSR challenge split (36 clips from Hugging Face, `SoccerNet/SN-GSR-2025`; the recorded share is gone), against a reference: the median of the other trainings' accepted fits on a frame one training's filter rejects.
    - **A gate came first:** the reference measured against GSR's truth camera on valid (`experiments/homography_joint/results/guard_part4_valid.json`). It covers 287 of 641 rejected frames; median 0.57 m, p90 1.92 m. The p90 exceeds the 1.0 m the harm margin needs.
    - **So the run was uninformative.** The guard stays out for good, and the challenge footage was not run.
    - **The guard's reconstruction** from per-frame logs equals the recorded strict runs on every frame.
    - **For the record, on valid** (development; the protocol's D1b p95, `guard_challenge.py descriptive`, `experiments/homography_joint/results/guard_part4_valid_descriptive.json`): it would have brought seed 4's worst clip from 27.58 to 0.48 m and seed 1's from 2.06 to 0.56 m.

## 8. TS-WorldCup: an external check

`experiments/homography_joint/ts_worldcup.py`, `experiments/homography_joint/results/ts_worldcup/`.

- **The data:** 43 broadcast sequences of the 2014 and 2018 World Cups, 3,812 frames with a homography each (Chu et al., CVsports 2022), not used to choose or confirm the frozen configuration; it did judge the guard (parts 1 to 3 of its rule) and removed it. Not part of the protocol's verdict.
- **The conversion:** the template is in yards, from image to template; it is converted to corner metres at 1080p. It is checked on a contact sheet and by white-line overlap: the truth beats the truth shifted by 2 m on 37 of 43 middle frames.
- **Scoring:** the frozen metrics, on the pitch-grid points the truth puts in view. This is not comparable with GSR's feet-based numbers; these are short, mostly near-goal shots.

| Row | track | AUC5 (grid) | p90 (m) | D1b p95 (m) |
|---|---|---|---|---|
| B0 | causal | 0.909 | 0.689 | 0.566 |
| B0 | final | 0.909 | 0.679 | 0.148 |
| P-final | causal | 0.954 | 0.340 | 0.110 |
| P-final | final | 0.954 | 0.338 | 0.037 |

- **Per sequence:** P-final's AUC5 is below B0's on 2 of 43.
- **At the 85 joined cuts** (first second after the cut):
  - P-final's `final` gives AUC5 0.947 and p90 0.46 m, as its causal track: the RTS pass does not smooth across a cut.
  - B0's `final` gives 0.813 and 2.71 m, against 0.901 and 1.18 m causal: its Savitzky-Golay pass smooths across the cut.

## 9. Opposite-side views

`VIEW_RULE.md`, `experiments/homography_joint/results/view_head.json`.

- **How often:** SN-Calib train has 540 opposite-side images and SN-Calib valid 50 (against 2,282 main). GSR valid has 7 such frames in 43,500.
- **The head:** a logistic view head on the frozen encoder.
- **It missed the bar fixed first** (recall ≥ 0.8 at ≤ 2 % false alarms on main views): recall 0.16 at 1.1 % false alarms, and 0.34 at 2 %. It is not shipped.

## 10. Limits

- **Opposite-side views** give a wrong matrix that is not flagged (section 9).
- **Single images along the pitch** with symmetric markings get no matrix. SN-Calib test availability is 0.771, against B0's 0.839.
- **The causal track depends on the training run** (section 7: over five seeds its worst valid clip ranges from 0.26 m to 27.6 m), and it jitters more than `final`. On the held-out run, causal D1b p95 is 0.175 m against B0's 0.473 m (all frames, `experiments/homography_bench/results/P-final/test/metrics.json`), while `final` gives 0.038 m. Use `final` where jitter matters.
- **A reset that is not a cut** moves the frames of its run of rejects to the new segment (SNGS-087: 9 frames worse by up to 9 m; AUC5 unchanged at three decimals).
- **The filter was tuned at 25 fps on every frame.** It steps in physical time; only strides 2 and 5 were measured.
- **The held-out run spent GSR test and SN-Calib test** for this line of work.

## 11. Where the records are

- **Code and records:** the local branch `research/homography-final` at `1492a5af7641`, not pushed.
  - Protocol and harness: `experiments/homography_bench/` (`METRICS.md`, `run.py`, `score.py`, `compare.py`; the scored records and comparisons in `experiments/homography_bench/results/`).
  - Hill climb: `experiments/homography_joint/` (`REPORT.md`, `FREEZE.md`, `HELDOUT.md`, the rules, the scripts, `experiments/homography_joint/results/`).
- **Weights and data** on this machine's storage, `/mnt/storage/tactifoot_research/homography_bench/`:
  - **primary weights:** `joint/package_runs/lines-r18s2w640/weights/final.pt`, sha256 `12f1ccccd1453154e4635cdc816a9587bacbdc8b15cbff39c122d76ce399ba43`, with `manifest.json` and the reconstructed `samples.jsonl` beside it. A copy is `models/pitch_lines_r18s2w640_tvtrain.{pt,json}` in the main checkout;
  - **seeds 1 to 4:** `joint/package_runs/lines-r18s2w640-seed{1,2,3,4}/`;
  - **the GSR challenge split:** `external/gsr_challenge/` (fetched for part 4 of the guard rule, not run);
  - **G's weights:** `joint/weights/r18s2w640.pt` (sha256 `8bdc510c…`);
  - **runs:** `runs/<row>/<split>/` (`run.json` gives commit and weights hash); TS-WorldCup runs in `joint/ts_worldcup/runs/`;
  - **TS-WorldCup data:** `external/ts_worldcup/`.

## 12. This change and the tested code

The tested code is the branch `feature/pitch-line-homography` at `e8b0326` (its code equals `research/homography-final` at `1492a5af7641`). It replaced the keypoint estimator in one change. Here the line estimator arrives in two:
- **this change** adds it beside the keypoint estimator, opt-in, under its own names;
- **the next change** makes it the default and deletes the keypoint estimator, `pitch/lines.py`, Savitzky-Golay and the pose backend. Its tree is made equal to the tested tree: `git diff e8b0326 -- src tests examples notebooks pyproject.toml uv.lock` is empty on its last commit.

**Byte-identical to the tested code** (`git diff e8b0326 -- <path>` is empty):
- `src/tactifoot_vision/pitch/fit.py`, `pitch/markings.py`, `pitch/temporal.py`: the joint fit, the markings, the Kalman filter and its RTS pass;
- `src/tactifoot_vision/models/_line_net.py`: the line network;
- `src/tactifoot_vision/data/pitch_lines.py`, `data/pitch_lines_v2.csv`: the line labels and the label rule;
- `src/tactifoot_vision/cli.py` and `src/tactifoot_vision/__init__.py`;
- `tests/line_points.py`, `tests/test_pitch_fit.py`, `tests/test_enums.py`.

**The tested code under this change's names.** These differ from the tested files only in the names `HomographyEstimator` -> `LineHomographyEstimator` and `smooth_homographies` -> `smooth_line_homographies`, and in the lines ruff wraps after the longer names:
- `src/tactifoot_vision/evaluation/lines.py`;
- `src/tactifoot_vision/models/lines.py`, the `lines` backend (one docstring reference);
- `tests/test_temporal.py`, `tests/test_pitch_lines.py` (one line, in a `model` test).

**The tested code in new glue.** Under those renames, the code is the tested code; around it, the glue is new:
- `pitch/line_homography.py` holds `HomographyFit` and `LineHomographyEstimator`. From `HomographyFit` to the end of the module it equals the tested `pitch/homography.py`. `frame_to_pitch`, `pitch_to_frame` and `visible_area` stay in `pitch/homography.py` beside the keypoint estimator.
- `pipeline/homographies.py` holds `frame_homography`, identical, and `smooth_line_homographies`, the tested `smooth_homographies` with one docstring sentence changed, beside the Savitzky-Golay `smooth_homographies`, which warns when it is given a line-model result.
- `Pipeline`, `FrameResult`, `PipelineResult`, the run file, the annotator, `Annotations`, `Dataset`, the enums and the registries take the tested additions beside what they had. `Pipeline` picks the line path only with a `line_model`, so a keypoint run takes the code it took before.
- The tests of the new behaviour are the tested ones, renamed, in new files (`tests/test_line_*.py`, `tests/test_smooth_line_homographies.py`), with tests of what only this change has (both models refused, a keypoint run unchanged). The tests of the keypoint path are unchanged; `tests/conftest.py` gains the `line_weights` fixture and `tests/test_enums.py` the count and the name rule of the new vocabularies.

**What was run here.** The fast suite and the linters, on a machine without a GPU, weights or data. The `model` tests, the CLI end to end, the notebooks and the frozen rows were not rerun for this change. The numbers above are those of the tested code, which the line path here runs unchanged.
