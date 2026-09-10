import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_contact_audit import sheet  # noqa: E402


def test_sheet_reuse_checks_shot_source_and_image(tmp_path):
    clip = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    assert writer.isOpened()
    for i in range(150):
        writer.write(np.full((48, 64, 3), i, dtype=np.uint8))
    writer.release()
    row = SimpleNamespace(review_id=0, shot_id="a", clip_path=str(clip))
    target = Path(sheet(row, tmp_path, 40))
    assert cv2.imread(str(target)).shape == (2430, 2400, 3)
    assert sheet(row, tmp_path, 40) == str(target)
    row.shot_id = "b"
    with pytest.raises(ValueError, match="identity changed"):
        sheet(row, tmp_path, 40)
    row.shot_id = "a"
    target.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="identity changed"):
        sheet(row, tmp_path, 40)
    target.unlink()
    sheet(row, tmp_path, 40)
    assert json.loads(target.with_suffix(".json").read_text())["identity"]["step_ms"] == 40
    with clip.open("ab") as stream:
        stream.write(b"changed source")
    with pytest.raises(ValueError, match="identity changed"):
        sheet(row, tmp_path, 40)


def test_legacy_sheet_requires_new_directory(tmp_path):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"source")
    (tmp_path / "000.jpg").write_bytes(b"old sheet")
    row = SimpleNamespace(review_id=0, shot_id="a", clip_path=str(clip))
    with pytest.raises(ValueError, match="Unverified"):
        sheet(row, tmp_path)
