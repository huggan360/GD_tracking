"""Kör separat efter träningen, i en miljö med Hailo Dataflow Compiler."""
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/yolo11n-products-demo/train/weights/best.pt"
DATA = ROOT / "data/products_synthetic_demo/dataset.yaml"

if __name__ == "__main__":
    # Ultralytics väljer YOLO11:s utgångar och sparar även metadata.yaml med klassnamnen.
    print(YOLO(str(WEIGHTS)).export(format="hailo", name="hailo8", data=str(DATA), imgsz=512))
