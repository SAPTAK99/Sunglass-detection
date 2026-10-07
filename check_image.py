import sys
from ultralytics import YOLO

img = sys.argv[1]
for name in ("SunglassV1", "SunglassV2", "SunglassV3"):
    r = YOLO(f"models/{name}.pt").predict(img, conf=0.01, verbose=False)[0]
    confs = [round(float(c), 2) for c in r.boxes.conf]
    print(name, "->", confs if confs else "no detection")