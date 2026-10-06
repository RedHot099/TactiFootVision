# Whole-video steps are post-processes on a saved result

Some steps need to see the whole video: lag-free homography smoothing, joining tracks into player tracks, and choosing the real ball among the ball candidates. These steps are functions that take a `PipelineResult` and return a new one. They are not a second pass inside `Pipeline.run`.

`Pipeline.run` stays causal: one pass, frame by frame.

A run file may list post-processes in its own section. Each entry maps to exactly one core function (ADR 0001), and every post-process is opt-in.

## Considered options

- **A second pass built into `Pipeline.run`.** Rejected. It would couple every whole-video step to the frame loop and make `Pipeline` grow with each new step. It would also force every step to rerun detection, when a saved result could be refined again and again.

## Consequences

- Post-processes work on saved run folders. A user can iterate on a post-process without re-running the models.
- A post-process that changes positions or identities must keep the result consistent: re-project `pitch_xy` from the new homographies, and remap `tracker_id`.
