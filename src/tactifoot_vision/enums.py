from enum import StrEnum


class DetectionBackend(StrEnum):
    FAKE = "fake"
    YOLO = "yolo"
    RFDETR = "rfdetr"
    RFDETR_SEG = "rfdetr_seg"


class DetectionTask(StrEnum):
    DETECT = "detect"
    SEGMENT = "segment"


class TrackingBackend(StrEnum):
    FAKE = "fake"
    BYTETRACK = "bytetrack"
    BOTSORT = "botsort"
    SAM2 = "sam2"


class KeypointBackend(StrEnum):
    NONE = "none"
    YOLO_POSE = "yolo_pose"


class DatasetFormat(StrEnum):
    YOLO = "yolo"
    COCO = "coco"


class DatasetSource(StrEnum):
    FILESYSTEM = "filesystem"
    SOCCERNET_TRACKING = "soccernet_tracking"
    SOCCERNET_GSR = "soccernet_gsr"


class Device(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class Sam2OutputBoxMode(StrEnum):
    MASK = "mask"
    DETECTOR = "detector"
    DETECTOR_STRICT = "detector_strict"
    DETECTOR_BLEND = "detector_blend"


class TeamAssignmentEmbedding(StrEnum):
    COLOR_HISTOGRAM = "color_histogram"
    RESNET = "resnet"
    SIGLIP = "siglip"


class TeamAssignmentReducer(StrEnum):
    NONE = "none"
    UMAP = "umap"


class TeamAssignmentClusterer(StrEnum):
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    CMEANS = "cmeans"


class TeamAssignmentCropMethod(StrEnum):
    CENTER = "center"
    OPENCV_MASK = "opencv_mask"
    SAM2_MASK = "sam2_mask"


class ExperimentKind(StrEnum):
    DETECTION_TRACKING = "detection_tracking"
    TEAM_CLASSIFICATION = "team_classification"
    FIRST5 = "first5"
    VIDEO_XG = "video_xg"
    HOMOGRAPHY_COMPARISON = "homography_comparison"


class HomographyMethod(StrEnum):
    CURRENT_YOLOPOSE_7PT = "current_yolopose_7pt"
    TVCALIB = "tvcalib"
    SPORTLIGHT = "sportlight"
    SOCCERSEGCAL = "soccersegcal"
    PNLCALIB = "pnlcalib"
    AUXFLOW = "auxflow"
    ORACLE_GSR_LINES_RANSAC = "oracle_gsr_lines_ransac"


class HomographyStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class BallReconstructionMethod(StrEnum):
    LINEAR = "linear"
    KALMAN_RTS = "kalman_rts"


class BallTrajectorySource(StrEnum):
    OBSERVED = "observed"
    INTERPOLATED = "interpolated"
    EXTRAPOLATED = "extrapolated"
    MISSING = "missing"


class ShotDetectorKind(StrEnum):
    METADATA = "metadata"
    KINEMATIC = "kinematic"


class VideoShotDetectorKind(StrEnum):
    CONTACT_KINEMATIC = "contact_kinematic"
    TEMPORAL_RANKER = "temporal_ranker"


class VideoDetectionVariant(StrEnum):
    LEGACY = "legacy"
    CHUNKED_BATCHED = "chunked_batched"
    MODEL_LADDER = "model_ladder"
    ADAPTIVE_SAMPLING = "adaptive_sampling"


class VideoBallReconstructionVariant(StrEnum):
    BASELINE_KALMAN = "baseline_kalman"
    VITERBI_DP = "viterbi_dp"
    KALMAN_RTS_V2 = "kalman_rts_v2"
    OPTICAL_FLOW_TEMPLATE = "optical_flow_template"


class VideoShotRankingVariant(StrEnum):
    BASELINE_CONTACT_KINEMATIC = "baseline_contact_kinematic"
    RULE_SWEEP = "rule_sweep"
    LEARNED_TEMPORAL = "learned_temporal"
    DENSE_LOCAL_REFINEMENT = "dense_local_refinement"
    HIGH_RECALL_CASCADE = "high_recall_cascade"
    HARD_NEGATIVE_CALIBRATED = "hard_negative_calibrated"
    WINDOWED_TEMPORAL = "windowed_temporal"


class VideoProjectionVariant(StrEnum):
    DEGRADED_IMAGE_NORMALIZED = "degraded_image_normalized"
    LAST_STABLE_HOMOGRAPHY = "last_stable_homography"
    LINE_BOX_HEURISTIC = "line_box_heuristic"
    QUALITY_AWARE_DEGRADED = "quality_aware_degraded"


class VideoXgCalibrationVariant(StrEnum):
    NONE = "none"
    COEFFICIENT_FIT = "coefficient_fit"
    ISOTONIC_PLATT = "isotonic_platt"
    QUALITY_AWARE_ENSEMBLE = "quality_aware_ensemble"
    NEURAL_VIDEO_XG = "neural_video_xg"
    DATABALLPY_SIMPLE_XG = "databallpy_simple_xg"


class XgModelKind(StrEnum):
    GEOMETRY = "geometry"
    VIDEO_GEOMETRY = "video_geometry"
    VIDEO_FREEZE_CONTEXT = "video_freeze_context"
    VIDEO_KINEMATIC_CONTEXT = "video_kinematic_context"


class ShotOutcome(StrEnum):
    UNKNOWN = "unknown"
    GOAL = "goal"
    ON_TARGET = "on_target"
    OFF_TARGET = "off_target"
    PENALTY = "penalty"
