from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CocoConversionReport:
    dataset_root: Path
    output_root: Path
    valid_fraction: float
    seed: int
    every_nth_frame: int
    sequences_total: int
    sequences_train: int
    sequences_valid: int
    test_split: str

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_root": str(self.dataset_root),
            "output_root": str(self.output_root),
            "valid_fraction": self.valid_fraction,
            "seed": self.seed,
            "every_nth_frame": self.every_nth_frame,
            "sequences_total": self.sequences_total,
            "sequences_train": self.sequences_train,
            "sequences_valid": self.sequences_valid,
            "test_split": self.test_split,
        }
