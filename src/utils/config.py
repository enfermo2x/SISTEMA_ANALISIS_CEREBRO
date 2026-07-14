import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

TRAINING_DIR = os.path.join(BASE_DIR, "Training")
TESTING_DIR = os.path.join(BASE_DIR, "Testing")

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_MAP = {cls: i for i, cls in enumerate(CLASSES)}
CLASS_MAP_INV = {i: cls for i, cls in enumerate(CLASSES)}

IMG_SIZE = (128, 128)
IMG_SIZE_150 = (150, 150)
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
SEED = 42
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
