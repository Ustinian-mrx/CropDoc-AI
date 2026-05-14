from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PLANT_VILLAGE_DIR = DATA_DIR / "PlantVillage"

MODELS_DIR = PROJECT_ROOT / "models"

IMAGE_SIZE = 224
NUM_CLASSES = 38
RANDOM_SEED = 42
