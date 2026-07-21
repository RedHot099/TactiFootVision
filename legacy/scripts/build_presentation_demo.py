import argparse
import ast
import json
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from sklearn.cluster import KMeans
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.loaders import load_config
from config.models import PitchVisualizerConfig
from tactifoot_vision.keypoints.yolo_pose_handler import YOLOPoseHandler
from tactifoot_vision.visualization.pitch_visualizer import PitchVisualizer


MATCH_TITLE = "LPP vs TBBN"
SOURCE_VIDEO_PATH = Path("/home/kuba/Downloads/lp_bbtn.mp4")
CLIP_START_SECONDS = 17 * 60 + 15
CLIP_DURATION_SECONDS = 60
CLIP_END_SECONDS = CLIP_START_SECONDS + CLIP_DURATION_SECONDS

ASSETS_DIR = project_root / "presentation_assets"
VIDEO_DIR = ASSETS_DIR / "video"
FRAME_DIR = ASSETS_DIR / "frames"
INTERMEDIATE_DIR = ASSETS_DIR / "intermediate"
MAIN_VIDEO_PATH = VIDEO_DIR / "pipeline_main_60s.mp4"
TEASER_VIDEO_PATH = VIDEO_DIR / "pipeline_teaser_12s.mp4"
TRIMMED_CLIP_PATH = VIDEO_DIR / "lp_bbtn_17m15s_18m15s_silent.mp4"
LAYOUT_NOTES_PATH = ASSETS_DIR / "layout_notes.md"
SLIDES_PLAN_PATH = ASSETS_DIR / "presentation_plan.md"
CONFIG_PATH = project_root / "config" / "presentation_demo_lp_bbtn.yaml"
PIPELINE_CSV_PATH = INTERMEDIATE_DIR / "pipeline_rfdetr_demo_pipelinedata_p1.csv"
PIPELINE_OUTPUT_VIDEO_STEM = INTERMEDIATE_DIR / "pipeline_rfdetr_demo.mp4"

CANVAS_W = 1920
CANVAS_H = 1080
TOP_BAR_H = 76
BOTTOM_BAR_H = 78
PANEL_PAD = 26
VIDEO_W = CANVAS_W
VIDEO_H = CANVAS_H - TOP_BAR_H - BOTTOM_BAR_H
INSET_W = 500
INSET_H = 360
INSET_MARGIN = 28

TEAM_0 = (235, 99, 37)
TEAM_1 = (60, 165, 255)
BALL = (245, 206, 66)
REFEREE = (210, 210, 210)
GOALKEEPER = (123, 227, 255)
TEXT = (240, 246, 252)
MUTED = (148, 163, 184)
BG = (10, 15, 23)
BG_2 = (15, 23, 32)
WHITE = (250, 250, 250)
BLACK = (10, 10, 10)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
SPACE_CONTROL_START_SECONDS = 30.0
SPACE_CONTROL_ALPHA_PITCH = 0.30
SPACE_CONTROL_ALPHA_FRAME = 0.22
SPACE_CONTROL_GRID_W = 240
SPACE_CONTROL_GRID_H = 160


@dataclass
class Observation:
    frame_id: int
    timestamp_seconds: float
    player_id: int
    type: str
    class_name: str
    location: Optional[np.ndarray]
    frame_bbox: Optional[np.ndarray]
    confidence: Optional[float]
    homography_matrix: Optional[np.ndarray]
    team_id: Optional[int]


def _parse_literal(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, (list, dict)):
        return value
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _ensure_dirs() -> None:
    for path in [ASSETS_DIR, VIDEO_DIR, FRAME_DIR, INTERMEDIATE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def trim_clip(force: bool = False) -> Path:
    if TRIMMED_CLIP_PATH.exists() and not force:
        return TRIMMED_CLIP_PATH
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(CLIP_START_SECONDS),
            "-i",
            str(SOURCE_VIDEO_PATH),
            "-t",
            str(CLIP_DURATION_SECONDS),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            str(TRIMMED_CLIP_PATH),
        ]
    )
    return TRIMMED_CLIP_PATH


def ensure_pipeline_csv(force: bool = False) -> Path:
    if PIPELINE_CSV_PATH.exists() and not force:
        return PIPELINE_CSV_PATH
    _run(
        [
            "uv",
            "run",
            "python",
            "scripts/run_detection.py",
            "--config",
            str(CONFIG_PATH),
        ],
        cwd=project_root,
    )
    if not PIPELINE_CSV_PATH.exists():
        raise FileNotFoundError(f"Expected pipeline CSV not found: {PIPELINE_CSV_PATH}")
    return PIPELINE_CSV_PATH


def load_observations(csv_path: Path) -> tuple[dict[int, list[Observation]], float, int]:
    df = pd.read_csv(csv_path)
    frame_map: dict[int, list[Observation]] = defaultdict(list)
    max_frame = 0
    fps_estimate = 25.0
    if "timestamp_seconds" in df.columns and df["timestamp_seconds"].notna().sum() > 5:
        timestamps = df[["frame_id", "timestamp_seconds"]].drop_duplicates().sort_values("frame_id")
        diffs = np.diff(timestamps["timestamp_seconds"].to_numpy(dtype=float))
        diffs = diffs[diffs > 0]
        if diffs.size:
            fps_estimate = float(round(1.0 / float(np.median(diffs)), 3))
    for row in df.itertuples(index=False):
        location_raw = _parse_literal(getattr(row, "location", None))
        bbox_raw = _parse_literal(getattr(row, "frame_bbox", None))
        homography_raw = _parse_literal(getattr(row, "homography_matrix", None))
        obs = Observation(
            frame_id=int(row.frame_id),
            timestamp_seconds=float(getattr(row, "timestamp_seconds", 0.0) or 0.0),
            player_id=int(getattr(row, "player_id", -1)),
            type=str(getattr(row, "type", "")),
            class_name=str(getattr(row, "class_name", "")),
            location=np.array(location_raw, dtype=np.float32) if location_raw is not None else None,
            frame_bbox=np.array(bbox_raw, dtype=np.float32) if bbox_raw is not None else None,
            confidence=float(getattr(row, "confidence")) if pd.notna(getattr(row, "confidence", np.nan)) else None,
            homography_matrix=np.array(homography_raw, dtype=np.float32) if homography_raw is not None else None,
            team_id=(
                int(getattr(row, "team_id"))
                if pd.notna(getattr(row, "team_id", np.nan))
                else None
            ),
        )
        frame_map[obs.frame_id].append(obs)
        max_frame = max(max_frame, obs.frame_id)
    return frame_map, fps_estimate, max_frame + 1


def _read_frame(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    return frame if ok else None


def _jersey_feature(frame: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 8 or y2 - y1 < 12:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    top = crop[int(ch * 0.12) : int(ch * 0.58), int(cw * 0.2) : int(cw * 0.8)]
    if top.size == 0:
        return None
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
    feat = np.concatenate([hist_h, hist_s]).astype(np.float32)
    denom = np.linalg.norm(feat)
    return feat / denom if denom > 0 else None


def infer_team_ids(video_path: Path, frame_map: dict[int, list[Observation]]) -> dict[int, int]:
    cap = cv2.VideoCapture(str(video_path))
    features_by_track: dict[int, list[np.ndarray]] = defaultdict(list)
    sampled_frames = sorted(frame_map.keys())[::10]
    for frame_idx in tqdm(sampled_frames[:220], desc="Inferring team colors"):
        frame = _read_frame(cap, frame_idx)
        if frame is None:
            continue
        for obs in frame_map.get(frame_idx, []):
            if obs.type != "player" or obs.player_id < 0 or obs.frame_bbox is None:
                continue
            feat = _jersey_feature(frame, obs.frame_bbox)
            if feat is not None:
                features_by_track[obs.player_id].append(feat)
    cap.release()
    track_ids = sorted(k for k, v in features_by_track.items() if len(v) >= 2)
    if len(track_ids) < 2:
        return {}
    x = np.stack([np.mean(features_by_track[tid], axis=0) for tid in track_ids], axis=0)
    model = KMeans(n_clusters=2, n_init=10, random_state=0)
    labels = model.fit_predict(x)
    return {track_id: int(label) for track_id, label in zip(track_ids, labels, strict=False)}


def canonicalize_team_sides(
    frame_map: dict[int, list[Observation]],
    inferred_teams: dict[int, int],
) -> dict[int, int]:
    team_locations: dict[int, list[float]] = defaultdict(list)
    for frame_idx in sorted(frame_map.keys())[:200]:
        for obs in frame_map.get(frame_idx, []):
            if obs.type != "player" or obs.location is None or obs.player_id < 0:
                continue
            raw_team = inferred_teams.get(obs.player_id)
            if raw_team is None:
                continue
            team_locations[int(raw_team)].append(float(obs.location[0]))
    if len(team_locations) < 2:
        return inferred_teams
    ordered = sorted(team_locations.items(), key=lambda item: float(np.mean(item[1])) if item[1] else 1e9)
    remap = {ordered[0][0]: 0, ordered[1][0]: 1}
    return {track_id: remap.get(team_id, team_id) for track_id, team_id in inferred_teams.items()}


def _team_for_obs(obs: Observation, inferred: dict[int, int]) -> Optional[int]:
    if obs.team_id is not None:
        return obs.team_id
    if obs.type == "player":
        return inferred.get(obs.player_id)
    return None


def _box_color(obs: Observation, inferred: dict[int, int]) -> tuple[int, int, int]:
    if obs.type == "ball":
        return BALL
    if obs.type == "referee":
        return REFEREE
    if obs.class_name == "goalkeeper":
        return GOALKEEPER
    team_id = _team_for_obs(obs, inferred)
    if team_id == 0:
        return TEAM_0
    if team_id == 1:
        return TEAM_1
    return WHITE


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.7,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _fit_panel(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
    canvas = np.full((target_h, target_w, 3), BG_2, dtype=np.uint8)
    y = (target_h - resized.shape[0]) // 2
    x = (target_w - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _draw_keypoints(frame: np.ndarray, keypoint_handler: YOLOPoseHandler) -> np.ndarray:
    result = keypoint_handler.detect(frame)
    if not result:
        return frame
    keypoints, _ = result
    if keypoints.xy.size == 0:
        return frame
    out = frame.copy()
    for xy, conf in zip(keypoints.xy[0], keypoints.confidence[0], strict=False):
        if float(conf) < 0.35:
            continue
        cx, cy = int(xy[0]), int(xy[1])
        cv2.circle(out, (cx, cy), 4, TEAM_0, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 7, WHITE, 1, lineType=cv2.LINE_AA)
    return out


def _make_pitch_visualizer() -> PitchVisualizer:
    return PitchVisualizer(
        PitchVisualizerConfig(
            enabled=True,
            pitch_color="#0f1720",
            line_color="#d5dde5",
            line_thickness=2,
            path_color="#ffff00",
            path_thickness=2,
            player_dot_radius=10,
            ball_dot_radius=7,
            player_color_default="#f8fafc",
            ball_color="#facc15",
            team_color_0="#2563eb",
            team_color_1="#f59e0b",
            canvas_width_px=INSET_W - 24,
            canvas_padding_px=18,
        ),
        pitch_dims=(PITCH_LENGTH, PITCH_WIDTH),
    )


def _current_homography(observations: list[Observation]) -> Optional[np.ndarray]:
    for obs in observations:
        if obs.homography_matrix is not None and obs.homography_matrix.shape == (3, 3):
            return obs.homography_matrix.astype(np.float32)
    return None


def _collect_space_control_points(
    observations: list[Observation], inferred_teams: dict[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    team_0: list[np.ndarray] = []
    team_1: list[np.ndarray] = []
    for obs in observations:
        if obs.location is None or obs.location.shape != (2,):
            continue
        if obs.type in {"ball", "referee"}:
            continue
        team_id = _team_for_obs(obs, inferred_teams)
        if team_id == 0:
            team_0.append(obs.location.astype(np.float32))
        elif team_id == 1:
            team_1.append(obs.location.astype(np.float32))
    team_0_arr = np.stack(team_0).astype(np.float32) if team_0 else np.empty((0, 2), dtype=np.float32)
    team_1_arr = np.stack(team_1).astype(np.float32) if team_1 else np.empty((0, 2), dtype=np.float32)
    return team_0_arr, team_1_arr


def _make_space_control_mask(
    team_0: np.ndarray,
    team_1: np.ndarray,
    width: int = SPACE_CONTROL_GRID_W,
    height: int = SPACE_CONTROL_GRID_H,
) -> Optional[np.ndarray]:
    if len(team_0) < 2 or len(team_1) < 2:
        return None
    xs = np.linspace(0.0, PITCH_LENGTH, width, dtype=np.float32)
    ys = np.linspace(0.0, PITCH_WIDTH, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid = np.stack([grid_x, grid_y], axis=-1)
    d0 = np.min(np.sum((grid[:, :, None, :] - team_0[None, None, :, :]) ** 2, axis=-1), axis=-1)
    d1 = np.min(np.sum((grid[:, :, None, :] - team_1[None, None, :, :]) ** 2, axis=-1), axis=-1)
    mask = np.zeros((height, width, 3), dtype=np.uint8)
    mask[d0 <= d1] = TEAM_0
    mask[d0 > d1] = TEAM_1
    return mask


def _blend_pitch_space_control(
    pitch: np.ndarray,
    pitch_visualizer: PitchVisualizer,
    space_mask: Optional[np.ndarray],
) -> np.ndarray:
    if space_mask is None:
        return pitch
    out = pitch.copy()
    inner_w = pitch_visualizer.canvas_width_px - 2 * pitch_visualizer.padding_px
    inner_h = pitch_visualizer.canvas_height_px - 2 * pitch_visualizer.padding_px
    resized = cv2.resize(space_mask, (inner_w, inner_h), interpolation=cv2.INTER_LINEAR)
    y1 = pitch_visualizer.padding_px
    y2 = y1 + inner_h
    x1 = pitch_visualizer.padding_px
    x2 = x1 + inner_w
    out[y1:y2, x1:x2] = cv2.addWeighted(
        out[y1:y2, x1:x2],
        1.0 - SPACE_CONTROL_ALPHA_PITCH,
        resized,
        SPACE_CONTROL_ALPHA_PITCH,
        0,
    )
    pitch_visualizer._draw_base_pitch(out)
    return out


def _logical_to_pitch_px(point: np.ndarray, pitch_visualizer: PitchVisualizer) -> Optional[tuple[int, int]]:
    return pitch_visualizer._scale_point(np.asarray(point, dtype=np.float32).reshape(1, 2))


def _draw_paths_on_pitch_cv(
    pitch: np.ndarray,
    pitch_visualizer: PitchVisualizer,
    paths: list[np.ndarray],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    for path in paths:
        if path.shape[0] < 2:
            continue
        scaled: list[tuple[int, int]] = []
        for point in path:
            scaled_point = _logical_to_pitch_px(point, pitch_visualizer)
            if scaled_point is not None:
                scaled.append(scaled_point)
        if len(scaled) >= 2:
            cv2.polylines(
                pitch,
                [np.array(scaled, dtype=np.int32)],
                False,
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )


def _draw_points_on_pitch_cv(
    pitch: np.ndarray,
    pitch_visualizer: PitchVisualizer,
    points: list[np.ndarray],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    for point in points:
        scaled_point = _logical_to_pitch_px(point, pitch_visualizer)
        if scaled_point is None:
            continue
        cv2.circle(pitch, scaled_point, radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(pitch, scaled_point, radius, BLACK, 2, lineType=cv2.LINE_AA)


def _apply_space_control_to_frame(
    frame: np.ndarray,
    observations: list[Observation],
    inferred_teams: dict[int, int],
    time_seconds: float,
) -> np.ndarray:
    if time_seconds < SPACE_CONTROL_START_SECONDS:
        return frame
    homography = _current_homography(observations)
    if homography is None:
        return frame
    team_0, team_1 = _collect_space_control_points(observations, inferred_teams)
    space_mask = _make_space_control_mask(team_0, team_1)
    if space_mask is None:
        return frame
    scale = np.array(
        [
            [(SPACE_CONTROL_GRID_W - 1) / PITCH_LENGTH, 0.0, 0.0],
            [0.0, (SPACE_CONTROL_GRID_H - 1) / PITCH_WIDTH, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    try:
        pitch_to_image = np.linalg.inv(homography).astype(np.float32)
    except np.linalg.LinAlgError:
        return frame
    warp_matrix = pitch_to_image @ np.linalg.inv(scale)
    warped = cv2.warpPerspective(
        space_mask,
        warp_matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    alpha_mask = np.full((SPACE_CONTROL_GRID_H, SPACE_CONTROL_GRID_W), 255, dtype=np.uint8)
    warped_alpha = cv2.warpPerspective(
        alpha_mask,
        warp_matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    if int(np.max(warped_alpha)) == 0:
        return frame
    alpha = ((warped_alpha.astype(np.float32) / 255.0) * SPACE_CONTROL_ALPHA_FRAME)[..., None]
    out = frame.astype(np.float32)
    out = out * (1.0 - alpha) + warped.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_left_panel(
    frame: np.ndarray,
    observations: list[Observation],
    inferred_teams: dict[int, int],
    time_seconds: float,
    show_boxes: bool,
    show_keypoints: bool,
    keypoint_handler: Optional[YOLOPoseHandler],
) -> np.ndarray:
    panel = frame.copy()
    panel = _apply_space_control_to_frame(panel, observations, inferred_teams, time_seconds)
    if show_keypoints and keypoint_handler is not None:
        panel = _draw_keypoints(panel, keypoint_handler)
    if show_boxes:
        for obs in observations:
            if obs.frame_bbox is None:
                continue
            color = _box_color(obs, inferred_teams)
            x1, y1, x2, y2 = obs.frame_bbox.astype(int)
            if obs.type == "ball":
                cx = (x1 + x2) // 2
                cy = max(12, y1 - 8)
                pts = np.array([[cx, cy - 10], [cx - 9, cy + 6], [cx + 9, cy + 6]])
                cv2.fillConvexPoly(panel, pts, color, lineType=cv2.LINE_AA)
                cv2.polylines(panel, [pts], True, BLACK, 2, lineType=cv2.LINE_AA)
                continue
            foot_x = (x1 + x2) // 2
            foot_y = y2
            width = max(12, x2 - x1)
            ellipse_axes = (width, max(5, int(0.35 * width)))
            cv2.ellipse(
                panel,
                (foot_x, foot_y),
                ellipse_axes,
                0,
                -45,
                235,
                color,
                2,
                lineType=cv2.LINE_4,
            )
            if obs.player_id >= 0:
                label = str(obs.player_id)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
                label_w = tw + 14
                label_h = th + 10
                label_x = int(foot_x - label_w / 2)
                label_y = max(0, int(foot_y - ellipse_axes[1] - label_h - 8))
                cv2.rectangle(
                    panel,
                    (label_x, label_y),
                    (label_x + label_w, label_y + label_h),
                    color,
                    -1,
                    lineType=cv2.LINE_AA,
                )
                _draw_text(
                    panel,
                    label,
                    (label_x + 7, label_y + th + 2),
                    scale=0.54,
                    color=BLACK,
                    thickness=2,
                )
    return panel


def _draw_pitch_panel(
    observations: list[Observation],
    inferred_teams: dict[int, int],
    trails: dict[int, deque[np.ndarray]],
    time_seconds: float,
    pitch_visualizer: PitchVisualizer,
) -> np.ndarray:
    pitch = pitch_visualizer.draw_frame()
    if pitch is None:
        pitch = np.full((480, 720, 3), BG_2, dtype=np.uint8)

    players_0: list[np.ndarray] = []
    players_1: list[np.ndarray] = []
    goalkeepers: list[np.ndarray] = []
    referees: list[np.ndarray] = []
    ball_xy: list[np.ndarray] = []
    team_paths_0: list[np.ndarray] = []
    team_paths_1: list[np.ndarray] = []
    neutral_paths: list[np.ndarray] = []

    for obs in observations:
        if obs.location is None or obs.location.shape != (2,):
            continue
        if obs.type == "ball":
            ball_xy.append(obs.location)
            continue
        if obs.type == "referee":
            referees.append(obs.location)
            if obs.player_id in trails and len(trails[obs.player_id]) >= 4:
                neutral_paths.append(np.array(trails[obs.player_id], dtype=np.float32))
            continue
        if obs.class_name == "goalkeeper":
            goalkeepers.append(obs.location)
            if obs.player_id in trails and len(trails[obs.player_id]) >= 4:
                neutral_paths.append(np.array(trails[obs.player_id], dtype=np.float32))
            continue
        team_id = _team_for_obs(obs, inferred_teams)
        if team_id == 0:
            players_0.append(obs.location)
            if obs.player_id in trails and len(trails[obs.player_id]) >= 4:
                team_paths_0.append(np.array(trails[obs.player_id], dtype=np.float32))
        elif team_id == 1:
            players_1.append(obs.location)
            if obs.player_id in trails and len(trails[obs.player_id]) >= 4:
                team_paths_1.append(np.array(trails[obs.player_id], dtype=np.float32))

    if time_seconds >= SPACE_CONTROL_START_SECONDS:
        team_0_arr, team_1_arr = _collect_space_control_points(observations, inferred_teams)
        pitch = _blend_pitch_space_control(
            pitch,
            pitch_visualizer,
            _make_space_control_mask(team_0_arr, team_1_arr),
        )

    for paths, color in [
        (team_paths_0, TEAM_0),
        (team_paths_1, TEAM_1),
        (neutral_paths, (212, 212, 216)),
    ]:
        if paths:
            _draw_paths_on_pitch_cv(pitch, pitch_visualizer, paths, color, thickness=2)

    if players_0:
        _draw_points_on_pitch_cv(pitch, pitch_visualizer, players_0, TEAM_0, radius=10)
    if players_1:
        _draw_points_on_pitch_cv(pitch, pitch_visualizer, players_1, TEAM_1, radius=10)
    if goalkeepers:
        _draw_points_on_pitch_cv(pitch, pitch_visualizer, goalkeepers, GOALKEEPER, radius=10)
    if referees:
        _draw_points_on_pitch_cv(pitch, pitch_visualizer, referees, REFEREE, radius=9)
    if ball_xy:
        _draw_points_on_pitch_cv(pitch, pitch_visualizer, ball_xy, BALL, radius=7)

    return cv2.resize(pitch, (INSET_W, INSET_H), interpolation=cv2.INTER_LINEAR)


def _make_metric_pill(label: str, value: str, width: int = 240) -> np.ndarray:
    pill = np.full((54, width, 3), BG_2, dtype=np.uint8)
    cv2.rectangle(pill, (0, 0), (width - 1, 53), (38, 52, 68), 1, lineType=cv2.LINE_AA)
    _draw_text(pill, label, (14, 22), scale=0.5, color=MUTED, thickness=1)
    _draw_text(pill, value, (14, 44), scale=0.72, color=TEXT, thickness=2)
    return pill


def render_demo(
    video_path: Path,
    frame_map: dict[int, list[Observation]],
    fps: float,
    total_frames: int,
    inferred_teams: dict[int, int],
    keypoint_handler: YOLOPoseHandler,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    pitch_visualizer = _make_pitch_visualizer()
    writer = cv2.VideoWriter(
        str(MAIN_VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (CANVAS_W, CANVAS_H),
    )
    trails: dict[int, deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=40))

    keyframe_targets = {
        "intro_split_screen.png": 3.0,
        "homography_keypoints.png": 5.0,
        "pitch_projection.png": 18.0,
        "final_trajectories.png": 58.0,
    }
    keyframe_frames = {name: int(round(seconds * fps)) for name, seconds in keyframe_targets.items()}

    for frame_idx in tqdm(range(total_frames), desc="Rendering presentation demo"):
        ok, frame = cap.read()
        if not ok:
            break
        obs = frame_map.get(frame_idx, [])
        timestamp = frame_idx / fps

        for item in obs:
            if item.player_id >= 0 and item.location is not None and item.location.shape == (2,):
                trails[item.player_id].append(item.location)

        show_boxes = timestamp >= 1.2
        show_keypoints = 3.0 <= timestamp < 6.0

        left_raw = _draw_left_panel(
            frame,
            obs,
            inferred_teams,
            time_seconds=timestamp,
            show_boxes=show_boxes,
            show_keypoints=show_keypoints,
            keypoint_handler=keypoint_handler,
        )
        video_panel = _fit_panel(left_raw, (VIDEO_W, VIDEO_H))
        inset_panel = _draw_pitch_panel(obs, inferred_teams, trails, timestamp, pitch_visualizer)

        canvas = np.full((CANVAS_H, CANVAS_W, 3), BG, dtype=np.uint8)
        canvas[TOP_BAR_H : CANVAS_H - BOTTOM_BAR_H, :VIDEO_W] = video_panel

        cv2.rectangle(canvas, (0, 0), (CANVAS_W, TOP_BAR_H), BG_2, -1)
        cv2.rectangle(canvas, (0, CANVAS_H - BOTTOM_BAR_H), (CANVAS_W, CANVAS_H), BG_2, -1)

        inset_x = CANVAS_W - INSET_W - INSET_MARGIN
        inset_y = CANVAS_H - BOTTOM_BAR_H - INSET_H - INSET_MARGIN
        canvas[inset_y : inset_y + INSET_H, inset_x : inset_x + INSET_W] = inset_panel

        _draw_text(canvas, MATCH_TITLE, (28, 46), scale=1.0, color=TEXT, thickness=2)
        _draw_text(
            canvas,
            "End-to-end football vision pipeline",
            (260, 46),
            scale=0.78,
            color=MUTED,
            thickness=2,
        )
        clip_clock = CLIP_START_SECONDS + frame_idx / fps
        minutes = int(clip_clock // 60)
        seconds = int(clip_clock % 60)
        _draw_text(canvas, f"Match time {minutes:02d}:{seconds:02d}", (1564, 46), scale=0.78, color=TEXT, thickness=2)

        if timestamp < 6.0:
            alpha = 1.0 if timestamp >= 0.5 else max(0.0, timestamp / 0.5)
            intro = canvas.copy()
            _draw_text(intro, "Video -> Detection -> Homography -> Pitch Analytics", (348, 116), scale=0.90, color=WHITE, thickness=2)
            canvas = cv2.addWeighted(intro, alpha, canvas, 1 - alpha, 0)

        tracked_players = sum(1 for item in obs if item.type == "player")
        ball_detected = any(item.type == "ball" for item in obs)
        homography_ok = any(item.homography_matrix is not None for item in obs)
        projected_points = sum(1 for item in obs if item.location is not None and item.type != "ball")

        pills = [
            _make_metric_pill("tracked players", str(tracked_players)),
            _make_metric_pill("ball detected", "yes" if ball_detected else "no"),
            _make_metric_pill("homography", "ok" if homography_ok else "pending"),
            _make_metric_pill("pitch projection", str(projected_points)),
        ]
        x = 28
        for pill in pills:
            canvas[CANVAS_H - BOTTOM_BAR_H + 12 : CANVAS_H - BOTTOM_BAR_H + 12 + pill.shape[0], x : x + pill.shape[1]] = pill
            x += pill.shape[1] + 18

        if frame_idx in keyframe_frames.values():
            for filename, target_idx in keyframe_frames.items():
                if target_idx == frame_idx:
                    cv2.imwrite(str(FRAME_DIR / filename), canvas)

        writer.write(canvas)

    cap.release()
    writer.release()


def transcode_main_video() -> None:
    temp_path = MAIN_VIDEO_PATH.with_name(f"{MAIN_VIDEO_PATH.stem}_h264{MAIN_VIDEO_PATH.suffix}")
    _run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(MAIN_VIDEO_PATH),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            str(temp_path),
        ]
    )
    temp_path.replace(MAIN_VIDEO_PATH)


def create_teaser_and_frames() -> None:
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            "6",
            "-i",
            str(MAIN_VIDEO_PATH),
            "-t",
            "12",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            str(TEASER_VIDEO_PATH),
        ]
    )


def write_layout_docs() -> None:
    layout_notes = f"""# Presentation Asset Notes

## Generated assets
- Main demo: `{MAIN_VIDEO_PATH}`
- Teaser: `{TEASER_VIDEO_PATH}`
- Trimmed clip: `{TRIMMED_CLIP_PATH}`
- Pipeline CSV: `{PIPELINE_CSV_PATH}`

## Recommended slide mapping
1. `intro_split_screen.png`
2. `intro_split_screen.png` + existing pipeline diagram from the paper
3. `homography_keypoints.png` + `pitch_projection.png`
4. Embed `{MAIN_VIDEO_PATH.name}`
5. `results/project/plots/first5/statsbomb360_mean_dist_and_coverage.png`
6. `results/project/plots/first5/map_vs_fps.png` and `articles/eccv/figures/quality_speed_tradeoff.png`
7. `final_trajectories.png` and team classification figure

## Demo source
- Source video: `{SOURCE_VIDEO_PATH}`
- Source segment: `17:15-18:15`
- Match title label: `{MATCH_TITLE}`
"""
    LAYOUT_NOTES_PATH.write_text(layout_notes, encoding="utf-8")

    slides_plan = f"""# 7-slide presentation plan

## Slide 1
Title: Od wideo meczu do analizy boiskowej: end-to-end football vision pipeline
Asset: `{FRAME_DIR / 'intro_split_screen.png'}`

## Slide 2
Asset stack:
- pipeline diagram from `articles/sdm/figs/process_imgs.png`
- raw frame / detections / pitch projection from generated keyframes

## Slide 3
Asset stack:
- `{FRAME_DIR / 'homography_keypoints.png'}`
- `{FRAME_DIR / 'pitch_projection.png'}`

## Slide 4
Embed:
- `{MAIN_VIDEO_PATH}`

## Slide 5
Charts:
- `results/project/plots/first5/statsbomb360_mean_dist_and_coverage.png`
- `results/project/plots/statsbomb360_eval/positional_error_ridgeplot.png`

## Slide 6
Charts:
- `results/project/plots/first5/map_vs_fps.png`
- `articles/eccv/figures/quality_speed_tradeoff.png`

## Slide 7
Asset stack:
- `{FRAME_DIR / 'final_trajectories.png'}`
- `results/team_classification/plots/plots_soccernet_tracking_structured_parquet/paper_embedding_siglip_vs_resnet.png`
"""
    SLIDES_PLAN_PATH.write_text(slides_plan, encoding="utf-8")


def main(force_trim: bool = False, force_pipeline: bool = False) -> None:
    _ensure_dirs()
    trim_clip(force=force_trim)
    pipeline_csv = ensure_pipeline_csv(force=force_pipeline)
    frame_map, fps_estimate, total_frames = load_observations(pipeline_csv)
    cfg = load_config(CONFIG_PATH)
    keypoint_handler = YOLOPoseHandler(cfg.keypoints, model_dir=(project_root / ".").resolve())
    inferred_teams = canonicalize_team_sides(
        frame_map,
        infer_team_ids(TRIMMED_CLIP_PATH, frame_map),
    )
    render_demo(
        TRIMMED_CLIP_PATH,
        frame_map,
        fps=fps_estimate,
        total_frames=total_frames,
        inferred_teams=inferred_teams,
        keypoint_handler=keypoint_handler,
    )
    transcode_main_video()
    create_teaser_and_frames()
    write_layout_docs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build presentation-ready demo assets.")
    parser.add_argument("--force-trim", action="store_true")
    parser.add_argument("--force-pipeline", action="store_true")
    args = parser.parse_args()
    main(force_trim=args.force_trim, force_pipeline=args.force_pipeline)
