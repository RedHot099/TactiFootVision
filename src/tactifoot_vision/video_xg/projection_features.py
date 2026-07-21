from pathlib import Path
from typing import Any

import pandas as pd


class HomographyArtifactProvider:
    def __init__(self, path: Path | None, *, max_age_seconds: float = 3.0) -> None:
        self.path = path
        self.max_age_seconds = max_age_seconds

    def project(self, sampled: pd.DataFrame) -> pd.DataFrame:
        if self.path is None or not self.path.exists():
            return _degraded(sampled, "degraded_image_normalized", 0.25)
        homographies = pd.read_parquet(self.path)
        if "global_frame_index" not in homographies.columns:
            return _degraded(sampled, "homography_schema_unsupported", 0.0)
        by_frame = homographies.set_index("global_frame_index")
        rows: list[dict[str, Any]] = []
        last: pd.Series | None = None
        last_seconds: float | None = None
        for sample in sampled.itertuples(index=False):
            if int(sample.global_frame_index) in by_frame.index:
                row = by_frame.loc[int(sample.global_frame_index)]
                if str(row.get("status", "available")) == "available":
                    last = row
                    last_seconds = float(sample.global_seconds)
            if (
                last is not None
                and last_seconds is not None
                and float(sample.global_seconds) - last_seconds <= self.max_age_seconds
            ):
                rows.append(
                    {
                        "global_frame_index": int(sample.global_frame_index),
                        "status": "last_stable_homography",
                        "projection_confidence": float(
                            last.get("projection_confidence", 0.75)
                        ),
                        "homography": last.get("homography", ""),
                    }
                )
            else:
                rows.append(
                    {
                        "global_frame_index": int(sample.global_frame_index),
                        "status": "homography_unavailable",
                        "projection_confidence": 0.0,
                        "homography": "",
                    }
                )
        return pd.DataFrame(rows)


class ImageLineHeuristicProjector:
    def project(self, sampled: pd.DataFrame) -> pd.DataFrame:
        if sampled.empty:
            return _degraded(sampled, "line_box_heuristic", 0.0)
        confidence = sampled.apply(
            lambda row: 0.35 if float(row["width"]) > float(row["height"]) else 0.2,
            axis=1,
        )
        frame = _degraded(sampled, "line_box_heuristic", 0.35)
        frame["projection_confidence"] = confidence.to_numpy()
        return frame


class ProjectionQualityAnnotator:
    def annotate(
        self, features: pd.DataFrame, homographies: pd.DataFrame
    ) -> pd.DataFrame:
        if features.empty:
            return features.copy()
        frame = features.drop(
            columns=["projection_status", "projection_confidence"],
            errors="ignore",
        ).copy()
        if homographies.empty:
            frame["projection_status"] = "degraded_image_normalized"
            frame["projection_confidence"] = 0.25
            return frame
        merged = frame.merge(
            homographies[["global_frame_index", "status", "projection_confidence"]],
            left_on="frame_index",
            right_on="global_frame_index",
            how="left",
        )
        merged["projection_status"] = merged["status"].fillna(
            "degraded_image_normalized"
        )
        merged["projection_confidence"] = merged["projection_confidence"].fillna(0.25)
        return merged.drop(columns=["global_frame_index", "status"])


def _degraded(sampled: pd.DataFrame, status: str, confidence: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "global_frame_index": int(row.global_frame_index),
                "status": status,
                "projection_confidence": confidence,
                "homography": "",
            }
            for row in sampled.itertuples(index=False)
        ]
    )
