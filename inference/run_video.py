import argparse
import time
from pathlib import Path

import cv2

from inference.detector import SunglassDetector
from utils.config import DEFAULT_CONF, DEFAULT_IMGSZ, MODEL_PATH, ROOT
from utils.draw import draw_detections, draw_fps
from utils.smoothing import TemporalFilter


def parse_args():
    p = argparse.ArgumentParser(description="Real-time sunglass detection")
    p.add_argument("--source", default="0", help="webcam index (0) or video file path")
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--output", default=None, help="output video path (overwrites if it exists)")
    p.add_argument("--save", action="store_true",
                help="auto-name the output: data/results/<video>_<model>_conf<value>.mp4")
    p.add_argument("--no-display", action="store_true", help="run without a window")
    return p.parse_args()


def main():
    args = parse_args()

    output = args.output
    if args.save and not output:
        out_dir = ROOT / "data" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{Path(str(args.source)).stem}_{Path(args.model).stem}_conf{args.conf}.mp4"
        output = str(out_dir / name)

    detector = SunglassDetector(args.model, args.conf, args.imgsz)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    writer = None
    if output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                                    fps_in, (w, h))
        print("Saving to:", output)

        tracker = TemporalFilter(start_conf=0.25, min_hits=2, max_misses=8)

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