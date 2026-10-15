# Drawing is stateless; frame history is passed explicitly

Some drawings need earlier frames: the ball trail, the radar's motion tails. `FrameAnnotator` and `PitchRadar` keep no memory between calls. A drawing that needs earlier frames takes them as an explicit `history` argument (the results of the preceding frames, oldest first), and `PipelineResult.window(video_index, seconds)` supplies that window. `render_video` uses the same window, so a notebook and a render draw the same image for the same frame.

## Considered options

- **Drawers that remember the frames they were called with.** Rejected. It is convenient in a loop, but drawing frames out of order, as a notebook does when it jumps to a frame, gives wrong trails. The same drawer object would also draw differently depending on what it drew before.
- **A drawing call that takes the whole result and a frame index.** Rejected. It changes the signature every caller uses today, for a need only the trail options have.

## Consequences

- Any frame of a saved result can be drawn, in any order, with its trail (ADR 0002).
- Each drawer filters `history` by its own trail length, so passing more frames than it needs is harmless.
