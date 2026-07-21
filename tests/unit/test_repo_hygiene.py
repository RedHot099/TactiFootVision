import subprocess
from pathlib import Path

REQUIRED_CONFIGS = (
    Path("configs/pipeline/fake_bytetrack.yaml"),
    Path("configs/pipeline/yolo_bytetrack_smoke.yaml"),
    Path("configs/experiments/team_classification_smoke.yaml"),
    Path("configs/experiments/soccernet_detection_tracking.yaml"),
)


def _git_check_ignore(path: Path) -> int:
    return subprocess.run(
        ["git", "check-ignore", "-q", str(path)],
        check=False,
    ).returncode


def test_root_config_yaml_remains_ignored() -> None:
    assert _git_check_ignore(Path("config.yaml")) == 0


def test_production_configs_are_not_ignored() -> None:
    assert _git_check_ignore(Path("configs/pipeline/fake_bytetrack.yaml")) != 0


def test_required_config_files_exist() -> None:
    missing = [str(path) for path in REQUIRED_CONFIGS if not path.is_file()]

    assert missing == []
