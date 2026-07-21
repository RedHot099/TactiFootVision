from pathlib import Path

from tactifoot_vision.shots import read_soccernet_action_metadata


def test_soccernet_action_position_maps_to_one_based_frame(tmp_path: Path) -> None:
    sequence = tmp_path / "SNMOT-001"
    sequence.mkdir()
    (sequence / "seqinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                "name=SNMOT-001",
                "frameRate=25",
                "seqLength=750",
                "imWidth=1920",
                "imHeight=1080",
                "imExt=.jpg",
            ]
        ),
        encoding="utf-8",
    )
    (sequence / "gameinfo.ini").write_text(
        "\n".join(
            [
                "[Sequence]",
                "name=SNMOT-001",
                "gameID=4",
                "actionPosition=505720",
                "actionClass=Shots on target",
                "clipStart=490000",
                "clipStop=520000",
            ]
        ),
        encoding="utf-8",
    )

    metadata = read_soccernet_action_metadata(sequence)

    assert metadata.action_frame == 394
    assert metadata.action_class == "Shots on target"
    assert metadata.game_id == "4"
