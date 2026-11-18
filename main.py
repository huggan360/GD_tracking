"""Edit the settings below, then run: python main.py"""

from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from utilities.devices import choose_device
from utilities.files import new_run_folder, write_json

ROOT = Path(__file__).resolve().parent

# Settings
# Use real scene boxes for fridge localization, or catalogue boxes for demos.
DATA = ROOT / "data/dataset.yaml"
WEIGHTS = ROOT / "models/pretrained/yolo26m.pt"
OUTPUT = ROOT / "models/grocery-v1"  # Use a new folder per run.
DEVICE = "auto"
SEED = 42
EPOCHS = 25
BATCH_SIZE = 8
IMAGE_SIZE = 640
WORKERS = 2
OPTIMIZER = "AdamW"
LEARNING_RATE = 0.0003
FINAL_LR_FRACTION = 0.1
WEIGHT_DECAY = 0.001
WARMUP_EPOCHS = 2
PATIENCE = 6
FREEZE_LAYERS = 10
AMP = True

# Data augmentation
# YOLO updates the product boxes together with each image transformation.
AUGMENTATION = True
ROTATION_DEGREES = 10
TRANSLATION = 0.1
SCALE = 0.2
HORIZONTAL_FLIP = 0.5
HUE = 0.015
SATURATION = 0.4
BRIGHTNESS = 0.3
MOSAIC = 0.0
MIXUP = 0.0


# Optional training without augmentation
class NoAugmentationTrainer(DetectionTrainer):
    """Turn off the entire augmentation pipeline, including Albumentations."""

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        dataset.augment = False
        dataset.transforms = dataset.build_transforms(self.args)
        if mode == "train":
            write_json(self.save_dir / "augmentation_policy.json", { "augmentation": False, "transforms": str(dataset.transforms), },)
        return dataset


def train():
    """Train one product box class; ViT supplies the grocery category."""
    if not DATA.is_file():
        raise ValueError(f"Missing detection dataset YAML: {DATA}")
    model = YOLO(str(WEIGHTS))
    if model.task != "detect":
        raise ValueError("Use detection weights such as yolo26m.pt.")
    output = new_run_folder(OUTPUT)

    # Training loop (Ultralytics)
    model.train(
        trainer=DetectionTrainer if AUGMENTATION else NoAugmentationTrainer,
        data=str(DATA),
        project=str(output),
        name="train",
        exist_ok=False,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        device=str(choose_device(DEVICE)),
        workers=WORKERS,
        optimizer=OPTIMIZER,
        lr0=LEARNING_RATE,
        lrf=FINAL_LR_FRACTION,
        weight_decay=WEIGHT_DECAY,
        cos_lr=True,
        warmup_epochs=WARMUP_EPOCHS,
        warmup_bias_lr=LEARNING_RATE,
        patience=PATIENCE,
        freeze=FREEZE_LAYERS,
        single_cls=True,
        seed=SEED,
        amp=AMP,
        degrees=ROTATION_DEGREES,
        translate=TRANSLATION,
        scale=SCALE,
        fliplr=HORIZONTAL_FLIP,
        flipud=0.0,
        hsv_h=HUE,
        hsv_s=SATURATION,
        hsv_v=BRIGHTNESS,
        mosaic=MOSAIC,
        mixup=MIXUP,
    )
    return Path(model.trainer.best)


# Run
if __name__ == "__main__":
    print(train())
