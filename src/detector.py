"""YOLO detector wrapper with normalized output and lazy local model loading."""
from pathlib import Path
import os
from typing import Optional


def _python_values(value):
    """Convert tensor, NumPy, or basic sequence values to ordinary Python values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value) if isinstance(value, (tuple, list)) else value


class ModelUnavailableError(RuntimeError):
    pass


def normalize_result(result) -> list[dict]:
    """Convert an Ultralytics Results object into simple Python records."""
    output = []
    if result is None or getattr(result, "boxes", None) is None:
        return output
    boxes = result.boxes
    names = getattr(result, "names", {}) or {}
    for i, box in enumerate(boxes):
        xyxy = _python_values(box.xyxy[0])
        class_id = int(box.cls[0].item())
        output.append({"xyxy": [float(v) for v in xyxy], "class_id": class_id,
                       "class_name": str(names.get(class_id, class_id)),
                       "confidence": float(box.conf[0].item()),
                       "track_id": int(box.id[0].item()) if getattr(box, "id", None) is not None else None})
    return output


class ObjectDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path or os.getenv("YOLO_MODEL_PATH", "models/yolo11n.pt"))
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model
        if not self.model_path.is_file():
            raise ModelUnavailableError(f"Model weights not found at {self.model_path}. Set YOLO_MODEL_PATH or add weights there.")
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
        except ImportError as exc:
            raise ModelUnavailableError("Ultralytics is not installed. Install requirements.txt first.") from exc
        return self._model

    def detect(self, image, confidence=0.25, classes=None):
        result = self.load().predict(source=image, conf=confidence, classes=classes, verbose=False)[0]
        return normalize_result(result)

    def track(self, frame, confidence=0.25, classes=None, tracker="bytetrack.yaml"):
        result = self.load().track(source=frame, conf=confidence, classes=classes, tracker=tracker,
                                   persist=True, verbose=False)[0]
        return normalize_result(result)
