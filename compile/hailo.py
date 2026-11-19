"""Kör med compile/.venv/bin/python compile/hailo.py efter träningen."""
import json
from pathlib import Path
import random

import cv2
from hailo_sdk_client import ClientRunner
import numpy as np
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import yaml

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/yolo11n-detection-demo/train/weights/best.pt"
DATA = ROOT / "data/products_detection_demo/dataset.yaml"
OUTPUT = WEIGHTS.parent / "best_hailo_model"
IMAGE_SIZE = 512
CALIBRATION_IMAGES = 1024

if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    build = ROOT / "compile/build"
    build.mkdir(parents=True, exist_ok=True)

    # Steg A: Exportera YOLO11 och välj de tre boxutgångarna och deras klasspoäng.
    model = YOLO(str(WEIGHTS))
    onnx = model.export(format="onnx", imgsz=IMAGE_SIZE, opset=11, simplify=True)
    head = len(model.model.model) - 1
    ends = [f"/model.{head}/cv{branch}.{scale}/cv{branch}.{scale}.2/Conv" for scale in range(3) for branch in (2, 3)]
    runner = ClientRunner(hw_arch="hailo8")
    runner.translate_onnx_model(str(onnx), "gd_yolo11n", end_node_names=ends)
    outputs = [layer.inputs[0].rsplit("/", 1)[-1] for layer in runner.get_hn_model().get_output_layers()]

    # Steg B: HailoRT gör om utgångarna till färdiga boxar och tar bort dubbletter.
    nms = {"nms_scores_th": 0.15, "nms_iou_th": 0.7, "image_dims": [IMAGE_SIZE, IMAGE_SIZE], "max_proposals_per_class": 100, "classes": len(model.names), "regression_length": 16, "background_removal": False, "background_removal_index": 0}
    nms["bbox_decoders"] = [{"name": f"bbox_decoder_{stride}", "stride": stride, "reg_layer": outputs[i * 2], "cls_layer": outputs[i * 2 + 1]} for i, stride in enumerate((8, 16, 32))]
    nms_file = OUTPUT / "nms_config.json"
    nms_file.write_text(json.dumps(nms, indent=2))
    script = ["normalization1 = normalization([0, 0, 0], [255, 255, 255])", "model_optimization_flavor(optimization_level=0)", "post_quantization_optimization(finetune, policy=disabled)"]
    script += [f"change_output_activation({outputs[i]}, sigmoid)" for i in (1, 3, 5)]
    script += [f'nms_postprocess("{nms_file}", meta_arch=yolov8, engine=cpu)', "allocator_param(width_splitter_defuse=disabled)"]
    runner.load_model_script("\n".join(script))

    # Steg C: Kalibrera INT8 med bilder från vår kylmiljö. Träna inte om vikterna.
    dataset = yaml.safe_load(DATA.read_text())
    images = sorted((Path(dataset["path"]) / dataset["val"]).glob("*.jpg"))
    selected = random.Random(42).sample(images, min(CALIBRATION_IMAGES, len(images)))
    letterbox = LetterBox(new_shape=(IMAGE_SIZE, IMAGE_SIZE), auto=False)
    calibration = np.stack([cv2.cvtColor(letterbox(image=cv2.imread(str(path))), cv2.COLOR_BGR2RGB) for path in selected])
    runner.optimize(calibration)
    runner.save_har(str(build / "yolo11n_quantized.har"))

    # Steg D: Kompilera till Hailo-8 och spara klassnamnet bredvid modellen.
    (OUTPUT / "best.hef").write_bytes(runner.compile())
    names = [model.names[i] for i in range(len(model.names))]
    (OUTPUT / "classes.txt").write_text("\n".join(names) + "\n")
    (OUTPUT / "metadata.yaml").write_text(yaml.safe_dump({"names": model.names, "imgsz": [IMAGE_SIZE, IMAGE_SIZE], "task": "detect", "hailo_arch": "hailo8", "nms": True}))
    print(OUTPUT / "best.hef", flush=True)
