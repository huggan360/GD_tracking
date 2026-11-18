"""Optional CPU model for local runs: python compile/export.py."""
import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/pi-demo/best.pt"
OUTPUT = WEIGHTS.parent / "yolo.pth"
IMAGE_SIZE = 640


def export():
    model = YOLO(str(WEIGHTS))
    if model.task != "detect":
        raise ValueError("Use YOLO26 detection weights.")
    path = model.export(format="torchscript", imgsz=IMAGE_SIZE, device="cpu", optimize=False)
    runtime = torch.jit.load(str(path), map_location="cpu")
    metadata = {"format": "gd-yolo26-v1", "classes": [model.names[i] for i in sorted(model.names)], "image_size": IMAGE_SIZE}
    torch.jit.save(runtime, str(OUTPUT), _extra_files={"metadata.json": json.dumps(metadata)})
    if Path(path) != OUTPUT:
        Path(path).unlink()
    print(OUTPUT)


if __name__ == "__main__":
    export()
