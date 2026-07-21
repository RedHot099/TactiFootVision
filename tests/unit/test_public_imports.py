import subprocess
import sys
import textwrap


def test_notebook_style_imports_do_not_load_heavy_model_modules() -> None:
    script = textwrap.dedent(
        """
        import sys

        from tactifoot_vision.datasets import SoccerNetTrackingDataset
        from tactifoot_vision.keypoints import YOLOPoseKeypointModel
        from tactifoot_vision.projection import PitchProjector
        from tactifoot_vision.team_assignment import TeamAssigner
        from tactifoot_vision.experiments import DetectionTrackingExperimentRunner

        _ = (
            SoccerNetTrackingDataset,
            YOLOPoseKeypointModel,
            PitchProjector,
            TeamAssigner,
            DetectionTrackingExperimentRunner,
        )
        loaded = [
            name
            for name in ("torch", "ultralytics", "transformers", "rfdetr", "cv2")
            if name in sys.modules
        ]
        if loaded:
            raise SystemExit(",".join(loaded))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout
