import numpy as np
from PIL import Image


def prepare_yolo_image(image, enabled=True):
    """Make edges clearer on a copy without moving or resizing any pixels."""
    rgb = image.convert("RGB")
    if not enabled:
        return rgb.copy()
    import cv2

    # Change brightness contrast, keeping colour channels separate.
    lab = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2LAB)
    light, a, b = cv2.split(lab)
    local_contrast = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    light = local_contrast.apply(light)

    # A small unsharp mask strengthens existing edges, not new outlines.
    blurred = cv2.GaussianBlur(light, (0, 0), sigmaX=1.0)
    light = cv2.addWeighted(light, 1.35, blurred, -0.35, 0)
    enhanced = cv2.cvtColor(cv2.merge((light, a, b)), cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced)
