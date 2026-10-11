import pandas as pd
from xgpoc.vendor_check.pairing_check import (
    assign_shots,
    classify_unpaired_shots,
    estimate_half_offset,
    recover_dropped_goals,
)


def test_offset_recovery_uses_densest_400_ms_window() -> None:
    soccer_net_ms = [10_100, 20_200, 30_300]
    statsbomb_ms = [10_000, 20_000, 30_000]

    assert estimate_half_offset(soccer_net_ms, statsbomb_ms) == 200.0


def test_assignment_is_one_to_one() -> None:
    soccer_net = pd.DataFrame(
        {"shot_id": ["sn-a", "sn-b"], "position_ms": [1_000, 1_400]}
    )
    statsbomb = pd.DataFrame(
        {"event_id": ["sb-a", "sb-b"], "timestamp_ms": [1_100, 2_000]}
    )

    pairs = assign_shots(soccer_net, statsbomb, offset_ms=0.0)

    assert pairs[["shot_id", "sb_event_id"]].to_dict("records") == [
        {"shot_id": "sn-a", "sb_event_id": "sb-a"},
        {"shot_id": "sn-b", "sb_event_id": "sb-b"},
    ]


def test_recovery_window_is_inclusive_at_both_ends() -> None:
    attempts = pd.DataFrame({"anchor_ms": [7_000, 28_000, 48_001]})
    goals = pd.DataFrame(
        {
            "event_id": ["early", "late", "outside"],
            "timestamp_ms": [10_000, 20_000, 40_000],
        }
    )

    recovered = recover_dropped_goals(attempts, goals, offset_ms=0.0)

    assert recovered[["anchor_ms", "sb_event_id"]].to_dict("records") == [
        {"anchor_ms": 7_000, "sb_event_id": "early"},
        {"anchor_ms": 28_000, "sb_event_id": "late"},
    ]


def test_population_rules_use_frozen_order() -> None:
    statsbomb = pd.DataFrame(
        {
            "event_id": [
                "recovered-penalty",
                "extra-time-penalty",
                "penalty-unaligned",
                "unaligned-near",
                "near-unpaired",
                "secondary",
            ],
            "period": [1, 3, 2, 2, 1, 1],
            "timestamp_ms": [1_000, 2_000, 3_000, 4_000, 10_000, 50_000],
            "shot_type": [
                "Penalty",
                "Penalty",
                "Penalty",
                "Open Play",
                "Open Play",
                "Open Play",
            ],
        }
    )
    offsets = {1: 1_000.0, 2: float("nan")}
    unpaired_soccer_net = pd.DataFrame({"half": [1, 2], "position_ms": [11_000, 5_000]})

    populations = classify_unpaired_shots(
        statsbomb,
        recovered_event_ids={"recovered-penalty"},
        offsets=offsets,
        unpaired_soccer_net=unpaired_soccer_net,
    )

    assert populations.set_index("event_id")["population"].to_dict() == {
        "recovered-penalty": "recovered_goal",
        "extra-time-penalty": "extra_time",
        "penalty-unaligned": "penalty",
        "unaligned-near": "unaligned_half",
        "near-unpaired": "possible_mispair",
        "secondary": "secondary",
    }
