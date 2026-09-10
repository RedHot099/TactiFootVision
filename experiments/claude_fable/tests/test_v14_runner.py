import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_v14 import apply_admissions  # noqa: E402


def _pool(tmp_path):
    frames = {}
    for shot in ("a", "b", "c", "d"):
        for offset in ("", "_-400", "_-800", "_-1200", "_-1600"):
            path = tmp_path / f"{shot}{offset or '_0'}.jpg"
            path.write_bytes(b"x")
            frames.setdefault(f"frame_path{offset}", []).append(str(path))
    return pd.DataFrame({"shot_id": list("abcd"), "role": ["fit", "fit", "select", "evaluate"], **frames})


def test_admissions_replace_only_the_input_frame_and_keep_every_record(tmp_path):
    pool = _pool(tmp_path)
    admissions = pd.DataFrame({"shot_id": ["a", "c", "d"], "admission": ["safe", "nominal", "unsafe"],
                               "frame_path": [pool.loc[0, "frame_path_-400"], pool.loc[2, "frame_path"], ""],
                               "gap_ms": [400, np.nan, -200]})
    out = apply_admissions(pool, admissions)
    assert len(out) == 4
    assert out.admission.tolist() == ["safe", "nominal", "nominal", "unsafe"]
    assert out.loc[0, "frame_path"] == pool.loc[0, "frame_path_-400"]
    assert out.loc[0, "frame_path_t0"] == pool.loc[0, "frame_path"]
    assert out.loc[1, "frame_path"] == pool.loc[1, "frame_path"]        # unreviewed: nominal t0
    assert out.loc[3, "frame_path"] == pool.loc[3, "frame_path_-1600"]  # unsafe: earliest frame, flagged
    assert out.loc[0, "gap_ms"] == 400


def test_admissions_must_match_the_pool_and_point_at_real_frames(tmp_path):
    pool = _pool(tmp_path)
    nominal_outside = pd.DataFrame({"shot_id": ["zz"], "admission": ["nominal"], "frame_path": [""], "gap_ms": [None]})
    assert apply_admissions(pool, nominal_outside).admission.eq("nominal").all()
    with pytest.raises(ValueError, match="not in the pool"):
        apply_admissions(pool, nominal_outside.assign(admission="safe"))
    with pytest.raises(ValueError, match="duplicate"):
        apply_admissions(pool, pd.DataFrame({"shot_id": ["a", "a"], "admission": ["nominal"] * 2, "frame_path": [""] * 2,
                                             "gap_ms": [None] * 2}))
    with pytest.raises(ValueError, match="frame missing"):
        apply_admissions(pool, pd.DataFrame({"shot_id": ["a"], "admission": ["safe"], "frame_path": ["/nope.jpg"],
                                             "gap_ms": [400]}))
