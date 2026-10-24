"""ball-detection 05: the rule's read of the fine-tune table (scripts/ball_finetune.py), on synthetic cards."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_finetune as bf  # noqa: E402


def card(low: float, people: bool = True, ekstraklasa: bool = True) -> dict:
    return {
        "rule": {
            "primary": {
                "difference": low + 0.01,
                "ci": [low, low + 0.02],
                "success": low > 0,
            },
            "guard_people": people,
            "guard_ekstraklasa": ekstraklasa,
            "ekstraklasa_ball_ap50_direction": False,
        }
    }


def test_the_winner_needs_the_last_epoch_b_minus_a_ci_above_zero_and_every_guard():
    assert bf.verdict({("B-A", 0): card(0.05), ("B", 0): card(0.05)}) == {
        "complete": False
    }
    won = bf.verdict({("B-A", 1): card(0.001), ("B", 1): card(-0.05)})
    assert won["winner"] == "B" and won["success"]
    assert (
        bf.verdict({("B-A", 1): card(-0.001), ("B", 1): card(0.05)})["winner"] is None
    )
    # guards against A (primary row) and against dd2c (the pipeline)
    assert (
        bf.verdict({("B-A", 1): card(0.01, people=False), ("B", 1): card(0.01)})[
            "winner"
        ]
        is None
    )
    assert (
        bf.verdict({("B-A", 1): card(0.01), ("B", 1): card(0.01, ekstraklasa=False)})[
            "winner"
        ]
        is None
    )
