"""Edit VIDEO and run: python track.py. This checks detection and tracking without ViT."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from utilities.files import new_run_folder, write_json
from utilities.preprocessing import prepare_yolo_image
from utilities.tracker import new_tracker

ROOT = Path(__file__).resolve().parent
VIDEO = ROOT / "data/clip.mp4"
WEIGHTS = ROOT / "models/pi-demo/best.pt"
OUTPUT = ROOT / "models/tracking-run"
BACKEND = "torch"  # "hailo" for a Pi with HailoRT installed.
HEF = ROOT / "models/pi-demo/yolo.hef"
LABELS = ROOT / "models/pi-demo/classes.txt"
SAMPLE_FPS = 5
CONFIDENCE = 0.15
PREPROCESSING = False
EXCLUDED = {"person", "refrigerator"}


def track():
    capture = cv2.VideoCapture(str(VIDEO))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or fps <= 0:
        raise ValueError(f"Cannot decode {VIDEO}")
    stride = max(1, round(fps / SAMPLE_FPS))
    tracker = new_tracker(fps / stride)
    if BACKEND == "hailo":
        from hailo.runtime import HailoDetector
        detector = HailoDetector(HEF, LABELS.read_text().splitlines(), CONFIDENCE)
        names = detector.names
    elif BACKEND == "torch":
        detector = YOLO(str(WEIGHTS))
        names = detector.names
    else:
        raise ValueError("BACKEND must be torch or hailo.")
    output = new_run_folder(OUTPUT)
    tracks, frame_index = {}, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current, frame_index = frame_index, frame_index + 1
            if current % stride:
                continue
            image = np.asarray(prepare_yolo_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), PREPROCESSING))
            if BACKEND == "hailo":
                rows = detector.detect(image)
            else:
                result = detector.predict(np.ascontiguousarray(image[:, :, ::-1]), imgsz=640, rect=False, conf=CONFIDENCE, verbose=False)[0]
                rows = result.boxes.data[:, :6].cpu().numpy()
            rows = rows[np.asarray([names[int(row[5])] not in EXCLUDED for row in rows], dtype=bool)]
            for x1, y1, x2, y2, identity, score, label, _ in tracker.update(Boxes(rows, frame.shape[:2]), image):
                key = str(int(identity))
                tracks.setdefault(key, {"id": key, "source_label": names[int(label)], "observations": []})
                tracks[key]["observations"].append({"frame": current, "time": current / fps, "box": [float(x1 / frame.shape[1]), float(y1 / frame.shape[0]), float(x2 / frame.shape[1]), float(y2 / frame.shape[0])], "detection_confidence": float(score)})
    finally:
        capture.release()
        if BACKEND == "hailo":
            from hailo.runtime import release_device
            release_device()
    write_json(output / "tracks.json", list(tracks.values()))
    return output


if __name__ == "__main__":
    print(track())
