# The homography is one joint fit to a line network's points, filtered by a plain Kalman filter

The package estimates each frame's homography from the painted pitch markings directly. A heatmap network (`lines`, `PitchLineModel`) finds points along the 20 SoccerNet markings. `HomographyEstimator` fits one robust homography to all of them: each point should lie on its marking's image line or circle. A constant-velocity Kalman filter then steadies the fits over time.

This replaces the pitch keypoint model as the source of the homography, the RANSAC fit to its 32 landmarks, the white-line mask and the chamfer refinement (`pitch/lines.py`), all deleted. It also replaces the Savitzky-Golay smoothing of homographies: the whole-video step `smooth_homographies` is now the filter's Rauch-Tung-Striebel (RTS) pass.

## The configuration (`GK-r18s2w640` in the homography hill climb; the package's defaults)

- **Line network r18s2w640.** ResNet-18 with an FPN decoder to stride 2, 640 x 360 input, 20.5 GMAC, 12.0 M parameters. It trains 20 epochs, batch 12, AdamW at 6e-4 with a cosine schedule, seed 0, on 16,866 images: the GSR train frames on disk and the 8,316 SN-Calib train images of the label rule.
- **The fit (G).** Levenberg-Marquardt on a soft-L1 loss, with the Jacobian in closed form. Its starts are the linear solve, the previous matrix and, only without one, three camera templates. Gating, horizon and uncertainty checks follow.
- **The cut restart.** When the fit started with the previous matrix is refused, the frame is solved again on its own, templates included. That fit is taken when its uncertainty is at most 1 m. The threshold is derived as half the rejection limit (`FitConfig.max_uncertainty_m / 2`), so it follows that limit.
- **The temporal filter.** The state is the pitch positions of four fixed image points and their rates. Acceleration noise is 0.15 m per step², measurement noise 0.3 m, in steps of 1/25 s. The gate is χ² 60, and the filter resets after 3 rejects in a row or 12 steps without a fit. A reset hands every frame since the run of rejects began to the new segment. A rejected frame keeps its own fit, and the next solve starts from it. (A rule that returned the prediction instead was tried and removed: see "Training-run variance".) The step follows the processed frame rate (`fps / stride`), so a strided or 50 fps run is filtered in the same physical time, and 25 fps without stride is one step per frame. The causal filter keeps no per-frame history.
  - `causal`: the per-frame path returns the filtered matrix.
  - `final`: `smooth_homographies` runs the RTS pass per segment, never across a reset.

## What the saved result carries

The RTS pass needs the estimator's matrices before the filter. `FrameResult.homography_fit` holds each frame's: `matrix` is the frame's own fit when `fresh`, else the held one. `FrameResult.homography` holds what the pass returned, which is filtered. The fresh fits are the filter's measurements. A frame outside every segment gets the matrix from before the filter: no fit yet, a long gap, or the rejected fits that a reset hands to the new view.

The post-process runs the same filter forward over those fits and then backward. It needs no state from the pass, only the fits and the filter's settings, which are its own arguments. A result without fits (a fixed homography, no line model) comes back unchanged.

## The held-out result: confirmed

The single held-out run of the frozen configuration (`docs/research/homography-hill-climb.md`, section 5; freeze commit `a6a0f1d` on the research branch) confirms it. On GSR test and SN-Calib test, the research branch's `compare.py B0 P-final --held-out` gives `beats_baseline = True`; every guardrail holds.

The package's behaviour is the one this run measured, on both tracks: a guard on rejected frames was added after it and removed again (see "Training-run variance"), and the four frozen rows rerun on valid and calib-valid after every later change to `src` equal their frozen records on every frame.

| `final` track, GSR test + SN-Calib test | B0 | P-final | difference [95 %] |
|---|---|---|---|
| AUC5 | 0.830 | **0.898** | +0.068 [+0.039, +0.107] |
| p90 (m) | 2.921 | **1.102** | −1.820 [−2.891, −0.896] |
| line accuracy@5 (mean of GSR and SN-Calib) | 0.485 | **0.642** | +0.157 [+0.138, +0.182] |
| jitter D1b p95 (m) | 0.185 | **0.039** | −0.147 [−0.285, −0.069] |
| cost per frame, GPU / CPU (ms) | 84.6 / 407 | **6.16 / 119** | |

- **These are the numbers to quote.**
- **Against valid:**
  - position error held: AUC5 0.886 on valid, 0.898 on test;
  - jitter was optimistic by about 15 % (0.033 m on valid, 0.038 m on test);
  - GSR line accuracy fell 5 points (0.867 to 0.815) while B0's rose 1 point, so the margin over B0 shrank from +21 to +14 points.
- **Per clip:** AUC5 is below B0's on 3 of 49 GSR test clips, by 0.007 at most.

## External check: TS-WorldCup (other tournaments and broadcasters)

This is the first set of numbers for this estimator outside SoccerNet: TS-WorldCup, 43 broadcast sequences of the 2014 and 2018 World Cups, 3,812 frames with a homography per frame (`docs/research/homography-hill-climb.md`, section 8).
- **How it is scored.** There are no people in the truth, so position error is that of the pitch-grid points the truth puts in view, scored by the frozen metrics. It is not comparable with GSR's feet-based numbers, and these short, near-goal sequences are easier.
- **Not part of the protocol's verdict.** Nothing was tuned on it.

| Row | track | AUC5 (grid) | p90 (m) | D1b p95 (m) | availability |
|---|---|---|---|---|---|
| B0 | causal | 0.909 | 0.689 | 0.566 | 1.000 |
| B0 | final | 0.909 | 0.679 | 0.148 | 1.000 |
| P-final | causal | 0.954 | 0.340 | 0.110 | 1.000 |
| P-final | final | 0.954 | 0.338 | 0.037 | 1.000 |

The estimator's AUC5 is below B0's on 2 of the 43 sequences.

## Numbers (development: GSR valid with its 20 cut pairs, SN-Calib valid)

Development numbers: the choices below were made on them. The held-out numbers above are the ones to quote.

| Row | track | AUC5 | p90 (m) | GSR acc@5 | D1b p95 (m) | gross | SN acc@5 | SN worst view @10 |
|---|---|---|---|---|---|---|---|---|
| B0 (keypoints + RANSAC + chamfer, Savitzky-Golay) | final | 0.819 | 2.428 | 0.661 | 0.264 | 1.2 % | 0.285 | 0.046 |
| G (the line fit alone; = F) | causal | 0.884 | 1.143 | 0.870 | 0.283 | 0.1 % | 0.469 | 0.222 |
| G + Savitzky-Golay | final | 0.884 | 1.143 | 0.869 | 0.068 | 0.1 % | 0.469 | 0.222 |
| **GK (declared: G + cut restart + filter)** | causal | 0.884 | 1.143 | 0.868 | 0.141 | 0.1 % | 0.469 | 0.222 |
| **GK** | final (RTS) | 0.885 | 1.140 | 0.870 | 0.032 | 0.1 % | 0.469 | 0.222 |
| **P-final (the package, weights trained through `tv.train`: the frozen primary row)** | causal | 0.885 | 1.142 | 0.865 | 0.139 | 0.1 % | 0.472 | 0.228 |
| **P-final** | final (RTS) | 0.886 | 1.139 | 0.867 | 0.033 | 0.1 % | 0.472 | 0.228 |
| P-final-gweights (the same code, G's weights) | final (RTS) | 0.885 | 1.141 | 0.870 | 0.033 | 0.1 % | 0.469 | 0.222 |
| P-final-nofilter (the filter off) | causal | 0.885 | 1.143 | 0.867 | 0.282 | 0.1 % | 0.472 | 0.228 |
| P-final-t1 (one template) | final (RTS) | 0.884 | 1.162 | 0.867 | 0.033 | 0.1 % | 0.461 | 0.201 |

- **Rows and commits.** P-final and its diagnostics run the package's API live from commit `2fd6084`. The earlier rows ran from the code of their own commits: P-lines at `510af71` on G's weights, and P-lines-retrain on the `tv.train` weights.
- **Against the earlier rows.** P-final's causal track equals P-lines-retrain's on every frame. Its `final` track differs on one frame. P-final-gweights differs from P-lines on 9 `final` frames of SNGS-087, which the fix to the filter's reset span moves.
- P-lines matches GK on every frame of both splits and both tracks, at most 7e-13 m apart.
- P-lines-raw matches G with the cut restart on every frame of its causal track.
- The research branch's `compare.py B0 P-final` gives `beats_baseline = True` on valid, with P-final's own cost records (window r4d): AUC5 +0.067, p90 −1.29 m, line accuracy +0.197, jitter −0.228 m, and every guardrail holds, cost, VRAM and seeds included. `compare.py B0 P-lines`, the commit before the review fixes, gave the same verdict. GK is the research code with the same output on every frame as P-lines; its own cost was not measured.
- Against G, GK's `final` is neutral on AUC5, p90 and lines, and improves jitter (0.068 → 0.032 m).

**The first second after a cut** (20 cut pairs, 474 truth frames, AUC5):

| | AUC5 |
|---|---|
| G (`causal`) | 0.832 |
| G + Savitzky-Golay (`final`), which smoothed across the cut | 0.740 |
| GK (`causal`) | 0.867 |
| GK (`final`) | 0.868 |
| the same second clips on their own (no previous view) | 0.867 |
| camera model ptz-J4 / flow flowJ5 (`final`) | 0.869 / 0.867 |

- The restart changes only those frames: no other frame of GSR valid or SN-Calib valid moves. It runs the template search on 132 instead of 74 of 43,500 GSR frames. The solver's mean time per frame stays at 2.80 ms; p99 goes from 7.52 to 7.66 ms (one thread, indicative).
- Cost of the package (P-final at `2fd6084`, window r4d, protocols `gpu-v2` / `cpu-v2`):
  - video: 6.16 ms per frame on the GPU (p95 8.9; VRAM 0.23 GB) and 119 ms on 4 CPU threads (p95 129). The owner's repeat on an idle machine (window r4e, background load at most 0.55) gave 118.1 ms (p95 127.2);
  - single images: 36.3 ms on the GPU (p95 58) and 129 ms on the CPU (p95 150), where the template search dominates;
  - P-lines, the same estimator before the review fixes (`a617c4f`), measured 6.07 ms (GPU) and 106 ms (CPU); the CPU difference is in the line network's time, not in the fit (`docs/research/homography-hill-climb.md`, section 6);
  - B0 under the same protocols: 84.6 ms (GPU) and 407 ms (CPU) per frame. G without the filter and the restart: 5.92 ms (GPU) and 118 ms (CPU).
- The filter's setting was chosen on all 58 clips by a rule committed before those runs (`docs/research/homography-hill-climb.md`, section 2): causal line accuracy at most 1 point below G, then the lowest causal jitter. Only accel 0.15 / meas 0.3 qualified. The control experiment's 0.05 / 0.3 costs 1.6 points of causal GSR line accuracy.
- The lower SN-Calib availability (P-final 0.774 on valid and 0.771 on test, against B0's 0.842 and 0.839; G's weights 0.773 on valid) is by decision: a fit the evidence leaves undetermined is refused, not guessed.

## Development numbers, not results

Three choices were made on valid:

- three camera templates;
- the restart's 1 m threshold, picked after looking at the five frames it changes;
- the filter's acceleration noise 0.15, which is the edge of the grid tried.

So every valid number above is a development number, jitter above all. The numbers to quote are those of the single held-out run of the frozen configuration (`docs/research/homography-hill-climb.md`, sections 2 and 5).

## Knowingly unsolved

- **Opposite-side views.** SoccerNet names lines in pitch coordinates, so a reverse-angle view's lines look like a main view of the other half with every name turned. Lines cannot tell them apart. They are left out of training. On one, the estimator fits the main view of the other half that it resembles: a wrong matrix it does not flag.
  - **How often.** The label rule finds 540 opposite-side images in SN-Calib train (against 12,116 main and 2,020 along). The oracle line fit finds 50 in SN-Calib valid (against 2,282 main and 200 along). GSR valid has 7 such frames in 43,500, in 2 clips.
  - **A flag from the image was tried** (`docs/research/homography-hill-climb.md`, section 9; rule and bar fixed first). A three-way logistic head (main, along, opposite) sat on the frozen encoder of the primary weights, pooled to keep the layout of stands and boards. It was trained on SN-Calib train, with the threshold set at 1 % out-of-fold false alarms on main views.
  - **It did not meet the bar** (recall ≥ 0.8 at ≤ 2 % false alarms on main views). On SN-Calib valid it flags 8 of 50 opposite images (recall 0.16), at 1.1 % false alarms on main views. The scores separate the views (AUC 0.92), but at 2 % false alarms recall is 0.34, and recall 0.8 costs about 15 %. So it is not shipped.
  - **What would be needed:** many more opposite-side images with labels than 540, a view head trained together with the encoder rather than on features frozen for line heatmaps, and, in video, a vote over a shot rather than per frame, since such views come in runs (replays, the reverse camera). An evaluation set larger than 50 images would be needed too, to state the recall to a few points.
- **Views along the pitch, as single images.** Behind or above a goal, the visible markings are often symmetric (the circle, the halfway line, a touchline). Without a previous matrix the fit is then undetermined and refused. In video the previous matrix decides; a single image gets no matrix.
- **A reset that is not a cut.** A reset hands every frame since the run of rejects began to the new segment, which is right for a real cut, a dissolve included. On valid the only reset with a gap inside its run of rejects is in SNGS-087, where the fits flip to a wrong view and back. There the 9 frames moved to the new segment score worse than the old segment's prediction did (0.1-0.4 m became 8-9 m on 5 of them). AUC5 does not move at three decimals.
- **The causal track depends on the training run.** A network can give, on a few clips, line points that support a wrong solution in single frames; the filter rejects such a fit but returns it, so it reaches the causal track. Over five training seeds of the declared recipe, causal D1b p95 on valid spreads by 0.47 m and the worst clip's by 27.3 m (seed 4 on SNGS-029: 27.6 m; seed 1 on SNGS-087: 2.06 m against 0.15 m with seed 0); position error and line accuracy do not move (final AUC5 spread 0.001). The `final` track repairs it (D1b spread 0.012 m). A guard that lets the frame's own points choose between the prediction and the fit cut the worst clips to 0.48 and 0.56 m on valid, but it was removed by the rule written for it and a last attempt was uninformative (see "Training-run variance"). Use the `final` track where jitter matters.
- **SNGS-036** stays below B0 (AUC5 −0.027), although its line accuracy is better than B0's.
- **The filter was tuned at 25 fps, every frame.** Its steps are in physical time, so other rates and strides use the same noise per second. Only stride 2 and 5 were measured, on valid. A frame between two processed frames holds the last matrix, which costs accuracy by itself. With physical time, causal AUC5 is 0.879 at stride 2 and 0.850 at stride 5; with one step per processed frame it is 0.877 and 0.838 (lines 0.764 against 0.751, and 0.516 against 0.457).

## Training-run variance

The recipe was trained four more times, with seeds 1 to 4 (development rows on valid; `docs/research/homography-hill-climb.md`, section 7).
- **Position error and line accuracy vary little over the five measured trainings:** over seeds 0 to 4, final AUC5 spread 0.001, p90 spread 3.4 mm, SN-Calib line accuracy@5 from 0.4656 to 0.4745.
- **Causal jitter does.** A network can give, on a few clips, line points that support a wrong solution in single frames. The filter rejects that fit but returns it as it is, so the jitter reaches the causal track. Seed 4 does this on SNGS-029 (worst clip causal D1b p95 27.6 m), seed 1 on SNGS-087 (2.06 m against 0.15 m), and G's weights on three test clips.
- **The `final` track repairs it:** D1b spread 0.012 m over the five seeds. The frozen configuration keeps this behaviour.
- **A guard was tried and removed** (`docs/research/homography-hill-climb.md`, section 7; its rule in three parts, each written before its runs). On a rejected frame before a reset, it returned the filter's prediction instead of the fit when the frame's own line points supported the prediction; `final` was identical by construction.
  - **Part 1** (majority support for the prediction) hid real cuts: a cut the old view still partly explains, and 20 frames of real changes on valid.
  - **Part 2** (the prediction only when it explains at least as many of the frame's points as the fit; amended so that a tie goes to the fit) was derived from every rejected frame on valid: a wrong single fit explains its own frame worse than the prediction (62 of 64), the first frames of a real change better (177 of 178, one tie). On valid it cut the seed spread of causal jitter from 0.039 to 0.009 m and of the worst clip from 1.80 to 0.31 m.
  - **Part 3** tested it on 85 real cuts joined from TS-WorldCup sequences, four sets of weights, guard on and off. At the cuts it changed nothing: in 166 decided post-cut frames per set of weights it never returned the prediction. Away from the cuts, for seed 0, jitter against TS-WorldCup's truth rose by 0.012 m [0.003, 0.024], which fails the rule's condition. That rise is one frame of one sequence, which the recipe puts into four cut clips; the rule counted those clips as independent. The rule said that failing removes the guard, so it was removed, and the frozen behaviour is the package's again.
  - **Part 4, the last attempt,** was to judge it on untouched footage (the GSR challenge split, fetched and left unrun) against a reference made from the other trainings' accepted fits. Its gate measured that reference on valid first: median 0.57 m and p90 1.92 m from GSR's truth camera, above the 1.0 m the rule needs to resolve its harm margin. The run was declared uninformative and the guard stays out. On valid, for the record, it would have brought the worst clips of seeds 4 and 1 from 27.6 and 2.06 m to 0.48 and 0.56 m.
  - **It has no held-out numbers.**

## Considered options

- **Keep the keypoint model and refine with lines** (B0). Rejected. It is worse on every quality number above, and it needs three parts (keypoints, RANSAC, a chamfer fit on a white-line mask) where one network and one fit do.
- **A symmetric fallback for symmetric evidence.** Removed. On valid it changed no frame's result: G reproduces F without it, bit for bit.
- **A camera model with pan-tilt-zoom parameters as the temporal layer** (ptz-J4). Rejected. It costs about 8 ms per frame and 2,500 lines for 22 % less causal jitter, loses 4 points of causal line accuracy on video, and is less steady on `final`.
- **Savitzky-Golay smoothing for `final`.** Removed. RTS gives D1b p95 0.032 m against its 0.068 m, at equal AUC5, p90 and line accuracy, and does not smooth across cuts.

## Consequences

- `Pipeline(line_model=...)` replaces `keypoint_model`, and the run-file section is `lines`. `FrameResult` keeps `line_points` and `homography_fit`. The annotator draws the line points, coloured by whether the homography puts them on their marking.
- `HomographyEstimator(temporal_filter=None)` turns the filter off. Its hold and template back-off are in seconds (`max_age_s`, `backoff_max_s`), counted in processed frames at the rate `reset` takes. `tv.pipeline.homographies.frame_homography` is the one per-frame step that the pipeline and `evaluate_lines` share. `smooth_homographies` works from the saved result alone (ADR 0002). The result records the filter settings its per-frame pass ran (`PipelineResult.temporal_filter`), and the pass uses them; an argument (`accel_m=...`) overrides one setting. A result made without a filter (`temporal_filter=None`) has nothing to run backward, so `smooth_homographies` keeps its per-frame homographies, with a warning, unless settings are given. `evaluate_lines` does the same: it scores the per-frame matrices of an estimator without a filter. This way the run-file path (`homography: {temporal_filter: ...}` with `postprocess: [{type: smooth_homographies}]`) and the evaluation agree: both score the backward pass of the filter that actually ran, or none.
- Every post-process stays opt-in (ADR 0002), so `tactifoot run` with the default run file gives the causal track. A user gets the `final` track with `postprocess: [{type: smooth_homographies}]` in the run file, or by calling `tv.pipeline.smooth_homographies(result)`; `examples/04_inference.py` does the latter.
- `tv.evaluation.evaluate_lines(model, clips)` scores trained line weights through the homography they give on labelled clips.
- The package ships no weights. `tv.train("lines", tv.load_pitch_lines(calibration, gamestate), epochs=20, batch_size=12)`, or `tactifoot train lines --calibration ... --gamestate ...`, trains them. Before the first epoch it writes the sample list (`samples.jsonl`) and a `manifest.json` of the recipe and the data. Training resumes per epoch, and only into a run folder with the same recipe and data.
- The `yolo_pose` backend, the pose datasets (YOLO `kpt_shape`, COCO keypoints), keypoint augmentation and keypoint evaluation are deleted too. After the port, nothing in the package, the examples, the notebooks or the CLI needed them, so they went in a commit of their own that can be reverted alone.
