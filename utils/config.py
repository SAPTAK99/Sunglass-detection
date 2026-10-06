from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "SunglassV1.pt"

DEFAULT_CONF = 0.4
DEFAULT_IMGSZ = 640
BOX_COLOR = (0, 200, 0)
TEXT_COLOR = (255, 255, 255)