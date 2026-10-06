import cv2
import numpy as np

from utils.config import BOX_COLOR, TEXT_COLOR


def draw_detections(frame: np.ndarray, detections) -> np.ndarray:
    for d in detections:
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), BOX_COLOR, 2)

        text = f"{d.label} {d.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(d.y1 - th - baseline, 0)

        cv2.rectangle(frame, (d.x1, top), (d.x1 + tw, top + th + baseline), BOX_COLOR, -1)
        cv2.putText(frame, text, (d.x1, top + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame