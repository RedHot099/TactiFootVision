from dataclasses import dataclass

from tactifoot_vision.domain import PitchPoint


@dataclass(frozen=True, slots=True)
class PitchModel:
    length: float = 105.0
    width: float = 68.0

    @property
    def keypoint_targets(self) -> dict[int, PitchPoint]:
        return {
            0: PitchPoint(0.0, 0.0),
            1: PitchPoint(self.length / 2.0, 0.0),
            2: PitchPoint(self.length, 0.0),
            3: PitchPoint(0.0, self.width),
            4: PitchPoint(self.length / 2.0, self.width),
            5: PitchPoint(self.length, self.width),
            6: PitchPoint(self.length / 2.0, self.width / 2.0),
        }
