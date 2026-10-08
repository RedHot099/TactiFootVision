"""Results visualisation from a saved pipeline result (run 04_inference.py first).

Run from the repository root: ``uv run python examples/05_visualization.py``
"""

from pathlib import Path

import tactifoot_vision as tv

OUT = Path("outputs/examples/05_visualization")
VIDEO = Path("data/videos/broadcast_60s.mp4")
OUT.mkdir(parents=True, exist_ok=True)

result = tv.PipelineResult.load("outputs/examples/04_inference/result.pkl")
frame_result = result[120]
frame = tv.VideoReader(VIDEO).read(frame_result.index)

standard = tv.viz.FrameAnnotator().annotate(frame, frame_result)
game = tv.viz.FrameAnnotator(style="video_game", draw_keypoints=False).annotate(
    frame, frame_result
)
radar = tv.viz.PitchRadar(draw_ids=True).draw(frame_result)
tv.show(
    [standard, game, radar], titles=["standard", "video game", "radar"], cols=1
).savefig(OUT / "frame_views.png")
tv.show(tv.viz.overlay(game, radar, position="bottom-right")).savefig(
    OUT / "overlay.png"
)

video_path = tv.viz.render_video(result, VIDEO, OUT / "annotated.mp4")
print("video:", video_path)

tv.viz.plot_heatmap(result, team=0).savefig(OUT / "heatmap_team0.png")
tv.viz.plot_heatmap(result, team=1).savefig(OUT / "heatmap_team1.png")
tv.viz.plot_tracks(result, max_tracks=8).savefig(OUT / "tracks.png")
print("figures in", OUT)
