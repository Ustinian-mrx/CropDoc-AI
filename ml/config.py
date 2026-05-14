from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PLANT_VILLAGE_DIR = DATA_DIR / "PlantVillage"

TRAIN_DIR = PLANT_VILLAGE_DIR / "train"
VAL_DIR = PLANT_VILLAGE_DIR / "val"

MODELS_DIR = PROJECT_ROOT / "models"

IMAGE_SIZE = 224
NUM_CLASSES = 38
BATCH_SIZE = 32
NUM_WORKERS = 0
RANDOM_SEED = 42

LEARNING_RATE = 1e-4
EPOCHS = 5
MODEL_NAME = "efficientnet_b0_plantvillage.pth"