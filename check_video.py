import sys

import cv2
from ultralytics import YOLO

video, model_path = sys.argv[1], sys.argv[2]
start = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
end = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0

model = YOLO(model_path)
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25

i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = i / fps
    i += 1
    if t < start:
        continue
    if t > end:
        break
    r = model.predict(frame, conf=0.05, verbose=False)[0]
    best = max((float(c) for c in r.boxes.conf), default=0.0)
    print(f"{t:5.2f}s  {best:.2f}  {'#' * int(best * 40)}")