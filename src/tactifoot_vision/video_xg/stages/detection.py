import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import pandas as pd

from tactifoot_vision.config.factories import build_detector
from tactifoot_vision.config.schemas import PipelineConfig
from tactifoot_vision.domain import DetectionSet, Frame
from tactifoot_vision.video_xg.artifacts import (
    read_dataframe_artifact,
    read_json_artifact,
    write_dataframe_artifact,
    write_json_artifact,
)
from tactifoot_vision.video_xg.config import VideoOnlyXgEndToEndConfig

DETECTION_COLUMNS = [
    "global_frame_index",
    "global_seconds",
    "part_index",
    "part_frame_index",
    "detection_index",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "width",
    "height",
]


class FrameDetector(Protocol):
    def predict(self, frame: Frame) -> DetectionSet:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DetectionChunkManifest:
    chunks: int
    rows: int
    frames: int
    chunk_size: int
    batch_size: int
    elapsed_seconds: float


class ChunkedDetectionRunner:
    def __init__(self, detector: FrameDetector | None = None) -> None:
        self.detector = detector

    def run(
        self,
        config: VideoOnlyXgEndToEndConfig,
        sampled: pd.DataFrame,
        output_dir: Path,
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        final_path = output_dir / "02_detections.parquet"
        chunks_dir = output_dir / "02_detections"
        manifest_path = chunks_dir / "manifest.json"
        if final_path.exists() and manifest_path.exists() and not force:
            return read_dataframe_artifact(final_path)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        detector = self.detector or build_detector(
            PipelineConfig(detection=config.detector)
        )
        chunk_paths = []
        for chunk_index, offset in enumerate(
            range(0, len(sampled), config.detection.chunk_size)
        ):
            chunk_path = chunks_dir / f"chunk_{chunk_index:05d}.parquet"
            chunk_paths.append(chunk_path)
            if chunk_path.exists() and not force:
                continue
            chunk = sampled.iloc[offset : offset + config.detection.chunk_size]
            frame = _detect_chunk(detector, chunk)
            write_dataframe_artifact(frame, chunk_path)
        merged = _merge_chunks(chunk_paths)
        write_dataframe_artifact(merged, final_path)
        manifest = DetectionChunkManifest(
            chunks=len(chunk_paths),
            rows=len(merged),
            frames=len(sampled),
            chunk_size=config.detection.chunk_size,
            batch_size=config.detection.batch_size,
            elapsed_seconds=time.perf_counter() - start,
        )
        write_json_artifact(asdict(manifest), manifest_path)
        return merged


class DetectionBenchmarkRunner:
    def __init__(self, detector: FrameDetector | None = None) -> None:
        self.detector = detector

    def run(
        self,
        config: VideoOnlyXgEndToEndConfig,
        sampled: pd.DataFrame,
        output_dir: Path,
    ) -> pd.DataFrame:
        limit = config.detection.benchmark_max_frames
        benchmark_sample = sampled if limit is None else sampled.head(limit)
        start = time.perf_counter()
        detections = ChunkedDetectionRunner(self.detector).run(
            config,
            benchmark_sample,
            output_dir,
            force=True,
        )
        elapsed = time.perf_counter() - start
        by_class = detections["class_name"].value_counts().to_dict()
        rows = [
            {
                "variant": config.detection.variant.value,
                "frames": float(len(benchmark_sample)),
                "detections": float(len(detections)),
                "elapsed_seconds": elapsed,
                "fps": float(len(benchmark_sample) / elapsed) if elapsed > 0.0 else 0.0,
                "ball_detections": float(by_class.get("ball", 0)),
                "player_detections": float(by_class.get("player", 0)),
            }
        ]
        frame = pd.DataFrame(rows)
        write_dataframe_artifact(frame, output_dir / "01_detection_benchmark.csv")
        return frame


def read_detection_manifest(path: Path) -> DetectionChunkManifest:
    data = read_json_artifact(path)
    return DetectionChunkManifest(
        chunks=int(data["chunks"]),
        rows=int(data["rows"]),
        frames=int(data["frames"]),
        chunk_size=int(data["chunk_size"]),
        batch_size=int(data["batch_size"]),
        elapsed_seconds=float(data["elapsed_seconds"]),
    )


def _detect_chunk(detector: FrameDetector, sampled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for video_path, group in sampled.groupby("video_path", sort=False):
        capture = cv2.VideoCapture(str(video_path))
        try:
            for sample in group.itertuples(index=False):
                image = _read_frame(capture, int(sample.part_frame_index))
                if image is None:
                    image = np.zeros(
                        (int(sample.height), int(sample.width), 3), dtype=np.uint8
                    )
                detections = detector.predict(
                    Frame(
                        index=int(sample.global_frame_index),
                        image=image,
                        timestamp_seconds=float(sample.global_seconds),
                        path=Path(str(video_path)),
                    )
                )
                for detection_index, detection in enumerate(detections):
                    rows.append(
                        {
                            "global_frame_index": int(sample.global_frame_index),
                            "global_seconds": float(sample.global_seconds),
                            "part_index": int(sample.part_index),
                            "part_frame_index": int(sample.part_frame_index),
                            "detection_index": detection_index,
                            "class_id": int(detection.class_id),
                            "class_name": detection.class_name,
                            "confidence": detection.confidence or 0.0,
                            "x1": detection.bbox.x1,
                            "y1": detection.bbox.y1,
                            "x2": detection.bbox.x2,
                            "y2": detection.bbox.y2,
                            "width": int(sample.width),
                            "height": int(sample.height),
                        }
                    )
        finally:
            capture.release()
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def _merge_chunks(paths: list[Path]) -> pd.DataFrame:
    frames = [read_dataframe_artifact(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame(columns=DETECTION_COLUMNS)
    merged = pd.concat(frames, ignore_index=True)
    return merged.sort_values(
        ["global_frame_index", "detection_index"], kind="stable"
    ).reset_index(drop=True)


def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, image = capture.read()
    if not ok:
        return None
    return np.asarray(image, dtype=np.uint8)
