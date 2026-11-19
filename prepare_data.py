from pathlib import Path

from utilities.synthetic_data import generate_dataset
from utilities.data import create_detection_dataset

ROOT = Path(__file__).resolve().parent

# Ändra här om vi vill bygga ett nytt dataset senare.
PRODUCTS = ROOT / "data/complete_dataset"
BACKGROUNDS = ROOT / "utilities/backgrounds"
OUTPUT = ROOT / "data/products_synthetic"
VARIANTS = 6
IMAGE_HEIGHT = 1024
WORKERS = 6
SEED = 42

if __name__ == "__main__":
    generate_dataset(PRODUCTS, BACKGROUNDS, OUTPUT, VARIANTS, IMAGE_HEIGHT, WORKERS, SEED)
    create_detection_dataset(OUTPUT, OUTPUT.with_name("products_detection_demo"), BACKGROUNDS, SEED)
