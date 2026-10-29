# Research summaries

Short HTML + PDF summaries of the research lines, meant for presenting the project.
They read results from the research worktrees (paths in `sources.py`) and write
everything to `outputs/research_summaries/<name>/`, which stays out of git.

xG from broadcast video (CPU only; about 15 min the first time, then cached):

```bash
cd reports/research_summaries
uv run python figures_xg.py                         # charts + numbers.json
CUDA_VISIBLE_DEVICES= uv run python media_xg.py     # shot stills, clip with a freeze at contact, Figure 2
uv run python build.py xg                           # xg_summary.html (self-contained) + .pdf
```

Ball localisation (CPU only, about 2 min; reads the ball-detection phase outputs under
`/mnt/storage/tactifoot_research/detector_data/ball/` (phases 1-3, tickets 01-23) and the 25 fps SoccerNet Tracking test frames):

```bash
uv run python figures_ball.py                       # error analysis, registered comparisons, breakdown, accuracy curve
uv run python media_ball.py                         # GSR test clip with a freeze + gallery crops
uv run python build.py ball                         # ball_summary.html + .pdf
```

Both reports share `summary.css`. The PDF is printed by headless Chromium; there the clip is replaced by its poster frame.
`geometry_xg.py` draws the measured inputs (cone, pressure, goalkeeper) from the research cache the
stacked model was fitted on, and checks them against the cached features.
Text and figure numbers are in `xg_summary.html`; the sources of every number are
the reports on `research/xg-detector` and `claude-fable/xg-poc`.
