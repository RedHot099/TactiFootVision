# Whole-video steps are post-processes on a saved result

Some steps need to see the whole video: lag-free homography smoothing, joining tracks into player tracks, and choosing the real ball among the ball candidates. These steps are functions that take a `PipelineResult` and return a new one. They are not a second pass inside `Pipeline.run`.

`Pipeline.run` makes one pass over the frames. It may end with a step that needs data only that pass has, but it never reads the video a second time.

A run file may list post-processes in its own section. Each entry maps to exactly one core function (ADR 0001), and every post-process is opt-in.

## Considered options

- **A second pass built into `Pipeline.run`.** Rejected. It would couple every whole-video step to the frame loop and make `Pipeline` grow with each new step. It would also force every step to rerun detection, when a saved result could be refined again and again.

## Consequences

- Post-processes work on saved run folders. A user can iterate on a post-process without re-running the models.
- A post-process that changes positions or identities must keep the result consistent: re-project `pitch_xy` from the new homographies, and remap `tracker_id`.

## Amendment: steps at the end of the pass

`Pipeline.run` ended with two steps that look at the whole run. This is how the code works today:

- **The team vote stays in `Pipeline.run`.** Each track gets one team by majority vote over its crops. The crops are sampled from the frames during the pass, and a saved `PipelineResult` holds no pixels. A post-process would have to read the video again. A step that needs data only the pass has belongs at the end of the pass.
- **The ball filter (`Pipeline._clean_ball`) moved into a post-process.** It dropped ball positions that jumped faster than `ball_max_speed` and needed only the result, which made it a post-process by this ADR. The `track_ball` post-process replaced it, together with `clean_ball_path` and `ball_max_speed`: it chooses the ball among the candidates, and its gate is the speed limit. `Pipeline.run` keeps the most confident candidate as the ball, so a homography post-process such as `reproject` has no ball filter to undo; `track_ball` runs after it.

Any step that can work on a saved result is a post-process.
