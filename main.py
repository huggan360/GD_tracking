"""Edit the settings below, then run: python main.py"""

from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from utilities.devices import choose_device
from utilities.files import new_run_folder
from utilities.plot import plot_loss

ROOT = Path(__file__).resolve().parent

# Settings
# Våra färdiggenererade kylbilder, med samma produktnamn som ViT.
DATA = ROOT / "data/products_synthetic/dataset.yaml"
WEIGHTS = ROOT / "models/pretrained/yolo11n.pt"
OUTPUT = ROOT / "models/yolo11n-products-v1"  # Ny mapp för varje träningskörning.
DEVICE = "auto"
SEED = 42
EPOCHS = 25
BATCH_SIZE = 2
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

# Bilderna är färdiga. Gör ingen extra augmentering när de läses in.
class ProductTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        dataset.augment = False
        dataset.transforms = dataset.build_transforms(self.args)
        return dataset

    # Spara loss-kurvan när epokens tränings- och valideringsvärden är klara.
    def save_metrics(self, metrics):
        super().save_metrics(metrics)
        plot_loss(self.csv, self.save_dir / "loss.png")


def train():
    """Träna YOLO11n på produktnamnen i det syntetiska datasetet."""
    if not DATA.is_file():
        raise ValueError(f"Missing detection dataset YAML: {DATA}")
    model = YOLO(str(WEIGHTS))
    if model.task != "detect":
        raise ValueError("Använd en detektionsmodell, exempelvis yolo11n.pt.")
    output = new_run_folder(OUTPUT)

    # Training loop (Ultralytics)
    model.train(
        trainer=ProductTrainer,
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
        single_cls=False,
        seed=SEED,
        amp=AMP,
        val=True,
        plots=False,
    )
    return Path(model.trainer.best)


# Run
if __name__ == "__main__":
    print(train())
