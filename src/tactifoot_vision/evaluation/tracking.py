from tactifoot_vision.domain import PipelineResult


def summarize_tracking(result: PipelineResult) -> dict[str, float]:
    track_ids = {track.track_id for frame in result.frames for track in frame.tracks}
    detections = sum(len(frame.tracks) for frame in result.frames)
    return {
        "frames": float(len(result.frames)),
        "tracks": float(len(track_ids)),
        "tracked_objects": float(detections),
    }
