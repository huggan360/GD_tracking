"""Edit the settings below, then run: python evaluate.py"""

from pathlib import Path

from ultralytics import YOLO

from utilities.devices import choose_device
from utilities.files import write_json

ROOT = Path(__file__).resolve().parent

# Settings
DATA = ROOT / "data/products_synthetic_demo/dataset.yaml"
CHECKPOINT = ROOT / "models/yolo11n-products-demo/train/weights/best.pt"
OUTPUT = ROOT / "models/yolo11n-products-demo"
SPLIT = "val"  # Syntetisk validering, inte nya verkliga produktfoton.
DEVICE = "auto"
BATCH_SIZE = 2
IMAGE_SIZE = 512
WORKERS = 2


# Load model and evaluate
def evaluate():
    if not CHECKPOINT.is_file():
        raise ValueError(f"Missing trained detector: {CHECKPOINT}")
    model = YOLO(str(CHECKPOINT))
    metrics = model.val(data=str(DATA), split=SPLIT, single_cls=False, imgsz=IMAGE_SIZE, batch=BATCH_SIZE, workers=WORKERS, device=str(choose_device(DEVICE)), project=str(OUTPUT), name=f"evaluation_{SPLIT}",)
    result = {key: float(value) for key, value in metrics.results_dict.items()}
    write_json(OUTPUT / f"evaluation_{SPLIT}.json", result)
    return result


# Run
if __name__ == "__main__":
    print(evaluate())
