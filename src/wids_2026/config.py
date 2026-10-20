"""Central configuration values for the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
DEFAULT_TARGET_COLUMN = "event"
DEFAULT_RANDOM_STATE = 42
