"""Locate the existing simulator without changing its files or environment."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "gym-two-car-reactive"
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))
ARTIFACTS = HERE / "artifacts"
MODELS = ARTIFACTS / "models"
DT = .01

