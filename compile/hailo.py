"""Edit these paths, then run: python compile/hailo.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hailo.compile_model import compile_model

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models/pi-demo/best.pt"
CALIBRATION = ROOT / "data/calibration"
OUTPUT = ROOT / "models/pi-demo/hailo"

if __name__ == "__main__":
    print(compile_model(WEIGHTS, CALIBRATION, OUTPUT))
