from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config.synloc_models import SynLocPrediction
from tactifoot_vision.synloc.submission import serialize_predictions
from sskit import image_to_ground

logger = logging.getLogger(__name__)


class LocSimCOCOeval(COCOeval):
    locsim_tau = 1.0

    def get_img_pos(self, detections: list[Mapping[str, object]]) -> list[np.ndarray]:
        keypoint_index = int(self.params.position_from_keypoint_index)
        positions: list[np.ndarray] = []
        for detection in detections:
            keypoints = np.asarray(detection["keypoints"], dtype=np.float32).reshape(-1, 3)
            positions.append(keypoints[keypoint_index, :2])
        return positions

    def computeIoU(self, imgId: int, catId: int):  # noqa: N802
        params = self.params
        if params.useCats:
            ground_truth = self._gts[imgId, catId]
            detections = self._dts[imgId, catId]
        else:
            ground_truth = [_ for c_id in params.catIds for _ in self._gts[imgId, c_id]]
            detections = [_ for c_id in params.catIds for _ in self._dts[imgId, c_id]]
        if len(ground_truth) == 0 or len(detections) == 0:
            return []

        indices = np.argsort([-det["score"] for det in detections], kind="mergesort")
        detections = [detections[index] for index in indices]
        if len(detections) > params.maxDets[-1]:
            detections = detections[: params.maxDets[-1]]

        image = self.cocoGt.loadImgs(int(imgId))[0]
        if hasattr(params, "position_from_keypoint_index"):
            image_points = np.asarray(self.get_img_pos(detections), dtype=np.float32)
            width = np.float32(image["width"])
            height = np.float32(image["height"])
            normalized = ((image_points - ((width - 1.0) / 2.0, (height - 1.0) / 2.0)) / width).astype(
                np.float32
            )
            det_pitch = np.asarray(
                image_to_ground(image["camera_matrix"], image["undist_poly"], normalized),
                dtype=np.float32,
            )[:, :2]
        else:
            det_pitch = np.asarray(
                [det["position_on_pitch"] for det in detections], dtype=np.float32
            )[:, :2]

        gt_pitch = np.asarray(
            [gt["position_on_pitch"] for gt in ground_truth], dtype=np.float32
        )[:, :2]

        aa, bb = np.meshgrid(gt_pitch[:, 0], det_pitch[:, 0])
        dist2 = (aa - bb) ** 2
        aa, bb = np.meshgrid(gt_pitch[:, 1], det_pitch[:, 1])
        dist2 += (aa - bb) ** 2
        return np.exp(np.log(0.05) * dist2 / self.locsim_tau**2)

    def accumulate(self, p=None):  # noqa: D401
        super().accumulate(p)
        params = self.params if p is None else p
        iou_mask = params.iouThrs == 0.5
        area_index = params.areaRngLbl.index("all")
        det_index = np.argmax(params.maxDets)

        precision = np.squeeze(self.eval["precision"][iou_mask, :, 0, area_index, det_index])
        scores = np.squeeze(self.eval["scores"][iou_mask, :, 0, area_index, det_index])
        recall = params.recThrs
        f1 = 2 * precision * recall / (precision + recall + np.spacing(1))

        self.eval["precision_50"] = precision
        self.eval["recall_50"] = recall
        self.eval["f1_50"] = f1
        self.eval["scores_50"] = scores

    def frame_accuracy(self, threshold: float) -> float:
        area_range = self.params.areaRng[self.params.areaRngLbl.index("all")]
        iou_mask = self.params.iouThrs == 0.5
        ok = 0
        bad = 0
        for eval_img in self.evalImgs:
            if eval_img is None or eval_img["aRng"] != area_range:
                continue
            matches = (eval_img["dtMatches"][iou_mask] > -1)[0]
            if (np.asarray(eval_img["dtScores"])[matches] > threshold).sum() == len(eval_img["gtIds"]):
                ok += 1
            else:
                bad += 1
        total = ok + bad
        return float(ok / total) if total else 0.0

    def summarize(self):  # noqa: D401
        super().summarize()
        if hasattr(self.params, "score_threshold"):
            threshold = float(self.params.score_threshold)
        else:
            scores = np.asarray(self.eval["scores_50"])
            f1 = np.asarray(self.eval["f1_50"])
            valid_scores = scores[scores >= 0]
            valid_f1 = f1[scores >= 0]
            if valid_scores.size == 0:
                threshold = 0.5
            else:
                index = int(np.argmax(valid_f1))
                if index + 1 < valid_scores.size:
                    threshold = float((valid_scores[index] + valid_scores[index + 1]) / 2.0)
                else:
                    threshold = max(0.0, float(valid_scores[index]) - 1e-6)

        scores = np.asarray(self.eval["scores_50"])
        index = int(np.searchsorted(-scores, -threshold, side="right") - 1)
        index = max(0, min(index, len(self.eval["precision_50"]) - 1))
        extra_stats = np.asarray(
            [
                float(self.eval["precision_50"][index]),
                float(self.eval["recall_50"][index]),
                float(self.eval["f1_50"][index]),
                float(threshold),
                float(self.frame_accuracy(threshold)),
            ],
            dtype=np.float32,
        )
        self.stats = np.concatenate([self.stats, extra_stats])


def evaluate_predictions(
    *,
    annotation_path: Path,
    predictions: Iterable[SynLocPrediction],
    score_threshold: float | None = None,
    position_from_keypoint_index: int | None = None,
) -> dict[str, float]:
    annotation_path = Path(annotation_path)
    coco = COCO(str(annotation_path))
    _normalize_coco_ground_truth(coco)
    coco.dataset.setdefault("info", {})
    coco.dataset.setdefault("licenses", [])
    results = serialize_predictions(
        predictions,
        position_from_keypoint_index=position_from_keypoint_index,
    )
    if not results:
        return _empty_stats_dict(score_threshold=score_threshold)
    coco_det = coco.loadRes(results)
    evaluator = LocSimCOCOeval(coco, coco_det, "bbox")
    evaluator.params.useSegm = None
    if score_threshold is not None:
        evaluator.params.score_threshold = float(score_threshold)
    if position_from_keypoint_index is not None:
        evaluator.params.position_from_keypoint_index = int(position_from_keypoint_index)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return _stats_to_dict(evaluator.stats)


def compare_evaluation_backends(
    *,
    annotation_path: Path,
    predictions: Iterable[SynLocPrediction],
    score_threshold: float | None = None,
    position_from_keypoint_index: int | None = None,
) -> dict[str, object]:
    local_metrics = evaluate_predictions(
        annotation_path=annotation_path,
        predictions=predictions,
        score_threshold=score_threshold,
        position_from_keypoint_index=position_from_keypoint_index,
    )
    reference_payload = _evaluate_with_reference_backend(
        annotation_path=annotation_path,
        predictions=predictions,
        score_threshold=score_threshold,
        position_from_keypoint_index=position_from_keypoint_index,
    )
    result: dict[str, object] = {
        "local": local_metrics,
        "reference_available": reference_payload is not None,
        "reference": None,
    }
    if reference_payload is None:
        result["reference_error"] = "Official reference backend unavailable (xtcocotools missing or sskit import failed)."
        return result

    result["reference"] = reference_payload
    result["delta"] = {
        name: float(local_metrics[name] - reference_payload[name])
        for name in local_metrics
        if name in reference_payload
    }
    return result


def _stats_to_dict(stats: np.ndarray) -> dict[str, float]:
    names = [
        "map_locsim",
        "ap_50",
        "ap_75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar_1",
        "ar_10",
        "ar_100",
        "ar_small",
        "ar_medium",
        "ar_large",
        "precision",
        "recall",
        "f1",
        "score_threshold",
        "frame_accuracy",
    ]
    return {name: float(value) for name, value in zip(names, stats)}


def _empty_stats_dict(*, score_threshold: float | None) -> dict[str, float]:
    names = [
        "map_locsim",
        "ap_50",
        "ap_75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar_1",
        "ar_10",
        "ar_100",
        "ar_small",
        "ar_medium",
        "ar_large",
        "precision",
        "recall",
        "f1",
        "score_threshold",
        "frame_accuracy",
    ]
    values = [0.0] * len(names)
    values[names.index("score_threshold")] = 0.5 if score_threshold is None else float(score_threshold)
    return {name: value for name, value in zip(names, values)}


def _normalize_coco_ground_truth(coco: COCO) -> None:
    annotations = coco.dataset.get("annotations", [])
    changed = False
    for annotation in annotations:
        if "iscrowd" not in annotation:
            annotation["iscrowd"] = 0
            changed = True
        if "area" not in annotation and "bbox" in annotation:
            x, y, w, h = [float(v) for v in annotation["bbox"]]
            annotation["area"] = float(max(0.0, w) * max(0.0, h))
            changed = True
    if changed:
        coco.createIndex()


def _load_reference_cocoeval():
    try:
        from xtcocotools.coco import COCO as XTCOCO  # type: ignore[import-not-found]
        from sskit.coco import LocSimCOCOeval as ReferenceLocSimCOCOeval  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - depends on optional backend
        logger.debug("Reference COCO backend unavailable: %s", exc)
        return None
    return XTCOCO, ReferenceLocSimCOCOeval


def _evaluate_with_reference_backend(
    *,
    annotation_path: Path,
    predictions: Iterable[SynLocPrediction],
    score_threshold: float | None = None,
    position_from_keypoint_index: int | None = None,
) -> dict[str, float] | None:
    backend = _load_reference_cocoeval()
    if backend is None:
        return None
    xtcoco_cls, evaluator_cls = backend
    coco = xtcoco_cls(str(annotation_path))
    coco.dataset.setdefault("info", {})
    coco.dataset.setdefault("licenses", [])
    results = serialize_predictions(
        predictions,
        position_from_keypoint_index=position_from_keypoint_index,
    )
    coco_det = coco.loadRes(results)
    evaluator = evaluator_cls(coco, coco_det, "bbox")
    evaluator.params.useSegm = None
    if score_threshold is not None:
        evaluator.params.score_threshold = float(score_threshold)
    if position_from_keypoint_index is not None:
        evaluator.params.position_from_keypoint_index = int(position_from_keypoint_index)
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return _stats_to_dict(np.asarray(evaluator.stats, dtype=np.float32))
