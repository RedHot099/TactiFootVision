from typing import Any

from tactifoot_vision.detection.adapters.rfdetr import RFDETRDetectionModel
from tactifoot_vision.domain import AdapterUnavailable


class RFDETRSegDetectionModel(RFDETRDetectionModel):
    model_name = "rfdetr_seg"

    def _model_class(self) -> type[Any]:
        import rfdetr

        if hasattr(rfdetr, "RFDETRSegPreview"):
            return rfdetr.RFDETRSegPreview
        if hasattr(rfdetr, "RFDETRSeg"):
            return rfdetr.RFDETRSeg
        raise AdapterUnavailable(
            "RF-DETR segmentation requires RFDETRSegPreview or RFDETRSeg."
        )
