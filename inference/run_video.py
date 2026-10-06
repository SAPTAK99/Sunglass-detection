import argparse
import time

import cv2

from utils.smoothing import TemporalFilter

from inference.detector import SunglassDetector
from utils.config import DEFAULT_CONF, DEFAULT_IMGSZ, MODEL_PATH
from utils.draw import draw_detections, draw_fps


def parse_args():
    p = argparse.ArgumentParser(description="Real-time sunglass detection")
    p.add_argument("--source", default="0", help="webcam index (0) or video file path")
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--output", default=None, help="optional output video path")
    p.add_argument("--no-display", action="store_true", help="run without a window")
    return p.parse_args()


def main():
    args = parse_args()
    detector = SunglassDetector(args.model, args.conf, args.imgsz)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    writer = None
    if args.output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                fps_in, (w, h))
    
    
    tracker = TemporalFilter()
    prev = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = tracker.update(detector.predict(frame))
            frame = draw_detections(frame, detections)

            now = time.perf_counter()
            frame = draw_fps(frame, 1.0 / max(now - prev, 1e-6))
            prev = now

            if writer:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Sunglass Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()