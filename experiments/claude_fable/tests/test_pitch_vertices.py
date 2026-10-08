"""Pins the 32 pitch landmarks the positional features fit homographies to.

The expected values were generated from the pre-rewrite
``tactifoot_vision.geometry.pitch_definitions.SoccerPitchConfiguration(105, 68)``
(git: 63db3fd^1). The rewritten package's ``SoccerPitch`` places keypoints
10, 11, 18, 19 at the penalty-arc intersection instead of goal-box width, so it
is NOT interchangeable: using it would change every earlier positional result.
"""

import numpy as np
from xgpoc.features.pitch import pitch_vertices

OLD_105x68 = np.array(
    [[0.0, 0.0], [0.0, 13.84000015258789], [0.0, 24.84000015258789], [0.0, 43.15999984741211],
     [0.0, 54.15999984741211], [0.0, 68.0], [5.5, 24.84000015258789], [5.5, 43.15999984741211],
     [11.0, 34.0], [16.5, 13.84000015258789], [16.5, 24.84000015258789], [16.5, 43.15999984741211],
     [16.5, 54.15999984741211], [52.5, 0.0], [52.5, 24.850000381469727], [52.5, 43.150001525878906],
     [52.5, 68.0], [88.5, 13.84000015258789], [88.5, 24.84000015258789], [88.5, 43.15999984741211],
     [88.5, 54.15999984741211], [94.0, 34.0], [99.5, 24.84000015258789], [99.5, 43.15999984741211],
     [105.0, 0.0], [105.0, 13.84000015258789], [105.0, 24.84000015258789], [105.0, 43.15999984741211],
     [105.0, 54.15999984741211], [105.0, 68.0], [43.349998474121094, 34.0], [61.650001525878906, 34.0]],
    dtype=np.float32,
)  # fmt: skip


def test_vertices_match_pre_rewrite_values():
    v = pitch_vertices(105.0, 68.0)
    assert v.dtype == np.float32
    np.testing.assert_array_equal(v, OLD_105x68)


def test_vertices_scale_with_pitch_size():
    v = pitch_vertices(120.0, 80.0)
    assert v.shape == (32, 2)
    assert v[29].tolist() == [120.0, 80.0]
    assert v[8, 0] == np.float32(11.0 / 105.0 * 120.0)
