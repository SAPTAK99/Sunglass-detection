from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from utils.config import DEFAULT_CONF, DEFAULT_IMGSZ, MODEL_PATH


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    label: str


class SunglassDetector:
    def __init__(self, model_path: Path = MODEL_PATH,
                 conf: float = DEFAULT_CONF, imgsz: int = DEFAULT_IMGSZ):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, verbose=False
        )[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            detections.append(Detection(
                x1, y1, x2, y2,
                confidence=float(box.conf[0]),
                label=result.names[cls_id],
            ))
        return detections