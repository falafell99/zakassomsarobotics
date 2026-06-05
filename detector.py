import logging
from schemas import Detection
from image_utils import normalize_bbox
from navigator import estimate_distance_from_bbox
import config

logger = logging.getLogger(__name__)

LABEL_ALIASES = {
    "tv": "monitor",
    "dining table": "desk",
    "cell phone": "phone",
    "potted plant": "plant",
    "couch": "sofa",
    "office_chair": "office chair",
    "coffee_table": "coffee table",
    "side_table": "side table",
    "floor_lamp": "floor lamp",
    "trash_can": "trash can",
    "storage_box": "storage box",
    "laundry_basket": "laundry basket",
    "moving_box": "moving box",
    "shoe_rack": "shoe rack",
    "staircase": "stairs",
    "steps": "stairs",
    "step": "stairs",
    "stair": "stairs",
    "handrail": "railing",
    "banister": "railing",
}

CATEGORIES = {
    "person": "human",
    "chair": "furniture",
    "office chair": "furniture",
    "sofa": "furniture",
    "desk": "furniture",
    "bed": "furniture",
    "bench": "furniture",
    "monitor": "office",
    "laptop": "office",
    "keyboard": "office",
    "mouse": "office",
    "phone": "office",
    "book": "office",
    "cup": "office",
    "bottle": "office",
    "plant": "decor",
    "trash can": "office",
    "printer": "office",
    "whiteboard": "office",
    "filing cabinet": "office",
    "wall": "structure",
    "door": "structure",
    "doorway": "structure",
    "stairs": "structure",
    "railing": "structure",
    "handrail": "structure",
    "backpack": "carry",
    "suitcase": "carry",
    "dog": "pet",
    "cat": "pet",
}


class ObjectDetector:
    def __init__(self):
        self.model = None
        self.names = {}
        self.device = None
        try:
            from ultralytics import YOLO
            self.model = YOLO(config.YOLO_MODEL)
            self.names = self.model.names
            self.device = self._choose_device()
            logger.info("YOLO detector loaded: %s", config.YOLO_MODEL)
            logger.info("YOLO device: %s", self.device or "auto")
        except Exception as exc:
            logger.warning("YOLO unavailable, detector fallback only: %s", exc)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _choose_device(self):
        if config.YOLO_DEVICE != "auto":
            return config.YOLO_DEVICE
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return None

    def detect(self, frame_bgr) -> list[Detection]:
        if self.model is None:
            return []

        height, width = frame_bgr.shape[:2]
        results = self.model.predict(
            source=frame_bgr,
            conf=config.DETECTION_CONFIDENCE,
            imgsz=config.YOLO_IMGSZ,
            max_det=config.YOLO_MAX_DETECTIONS,
            device=self.device,
            agnostic_nms=True,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                raw_label = str(self.names.get(cls_id, cls_id)).lower()
                label = LABEL_ALIASES.get(raw_label, raw_label)
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                bbox = normalize_bbox((x1, y1, x2, y2), width, height)
                distance = estimate_distance_from_bbox(label, bbox)
                detections.append(
                    Detection(
                        label=label,
                        raw_label=raw_label,
                        category=CATEGORIES.get(label, "object"),
                        confidence=round(conf, 3),
                        distance=distance,
                        bbox=bbox,
                        source="yolo",
                    )
                )
        return detections


detector = ObjectDetector()
