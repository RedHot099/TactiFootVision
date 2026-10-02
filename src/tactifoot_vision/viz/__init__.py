"""Results visualisation: annotated frames, pitch radar, videos and notebook plots.

    import tactifoot_vision as tv

    image = tv.viz.FrameAnnotator(style="video_game").annotate(frame, result[0])
    image = tv.viz.overlay(image, tv.viz.PitchRadar().draw(result[0]))
    tv.viz.render_video(result, "match.mp4", "annotated.mp4")
    tv.viz.plot_heatmap(result, team=0)

OpenCV drawing returns BGR ``uint8`` images; the plotting helpers return
matplotlib figures (matplotlib is imported on first use).
"""

from tactifoot_vision.viz.annotate import FrameAnnotator, draw_annotations
from tactifoot_vision.viz.plots import (
    draw_pitch,
    plot_distance_histogram,
    plot_heatmap,
    plot_metrics,
    plot_tracks,
    plot_training,
    show,
    show_augmentations,
    show_samples,
)
from tactifoot_vision.viz.radar import PitchRadar, overlay
from tactifoot_vision.viz.video import render_video

__all__ = [
    "FrameAnnotator",
    "PitchRadar",
    "draw_annotations",
    "draw_pitch",
    "overlay",
    "plot_distance_histogram",
    "plot_heatmap",
    "plot_metrics",
    "plot_tracks",
    "plot_training",
    "render_video",
    "show",
    "show_augmentations",
    "show_samples",
]
