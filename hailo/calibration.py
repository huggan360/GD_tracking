"""Make calibration images without changing the trained weights."""
from pathlib import Path

import cv2
import numpy as np
from hailo.runtime import letterbox
from utilities.preprocessing import prepare_yolo_image
from PIL import Image


def calibration_images(directory, limit=64, preprocessing=False):
    paths = sorted(path for path in Path(directory).rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise ValueError(f"Add representative fridge images to {directory} before compiling.")
    selected = np.linspace(0, len(paths) - 1, min(limit, len(paths)), dtype=int)
    images = []
    for index in selected:
        with Image.open(paths[index]) as image:
            rgb = np.asarray(prepare_yolo_image(image, preprocessing))
        images.append(letterbox(rgb, 640, 640)[0])
    return np.stack(images)
