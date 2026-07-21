import configparser
from dataclasses import dataclass
from pathlib import Path

from tactifoot_vision.datasets.soccernet_tracking import read_seqinfo


@dataclass(frozen=True, slots=True)
class SoccerNetActionMetadata:
    sequence_name: str
    game_id: str
    action_position_ms: int
    action_class: str
    clip_start_ms: int
    clip_stop_ms: int
    frame_rate: int
    seq_length: int

    @property
    def action_frame(self) -> int:
        frame = round(
            ((self.action_position_ms - self.clip_start_ms) / 1000.0) * self.frame_rate
        )
        return min(max(1, frame + 1), self.seq_length)


def read_soccernet_action_metadata(sequence_dir: str | Path) -> SoccerNetActionMetadata:
    path = Path(sequence_dir)
    gameinfo_path = path / "gameinfo.ini"
    parser = configparser.ConfigParser()
    parser.read(gameinfo_path)
    if "Sequence" not in parser:
        raise ValueError(f"Invalid gameinfo.ini (missing [Sequence]): {gameinfo_path}")
    section = parser["Sequence"]
    seqinfo = read_seqinfo(path)
    return SoccerNetActionMetadata(
        sequence_name=section.get("name", fallback=path.name).strip() or path.name,
        game_id=section.get("gameID", fallback="").strip(),
        action_position_ms=int(section["actionPosition"]),
        action_class=section.get("actionClass", fallback="").strip(),
        clip_start_ms=int(section["clipStart"]),
        clip_stop_ms=int(section["clipStop"]),
        frame_rate=seqinfo.frame_rate,
        seq_length=seqinfo.seq_length,
    )
