"""
config.py
---------
Proje genelinde kullanilan sabit degerler ve hyperparametreler.

Tek noktadan konfigurasyon yonetimi (Single Source of Truth) prensibi.
"""

from pathlib import Path

# ------------------------- Dizinler -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"        # Veri seti buraya indirilecek
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

for d in [RESULTS_DIR, CHECKPOINT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ------------------------- Veri Seti -------------------------
# Kaynak: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
# Brain Tumor MRI Dataset (Masoud Nickparvar) - 4 sinif, ~7023 gorsel
DATASET_NAME = "Brain Tumor MRI Classification"
DATASET_URL = "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)

# ------------------------- Egitim -------------------------
IMG_SIZE = 224          # Transfer learning icin standart giris boyutu
BATCH_SIZE = 32
EPOCHS = 50
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5
VALIDATION_SPLIT = 0.15  # Training icinden ayrilacak orani
RANDOM_SEED = 42

# ------------------------- Callbacks -------------------------
EARLY_STOPPING_PATIENCE = 8
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

# ------------------------- Data Augmentation -------------------------
AUG_ROTATION = 15
AUG_WIDTH_SHIFT = 0.1
AUG_HEIGHT_SHIFT = 0.1
AUG_ZOOM = 0.1
AUG_HORIZONTAL_FLIP = True

# ------------------------- Modeller -------------------------
MODEL_NAMES = ["CustomCNN", "ResNet50", "EfficientNetB0", "DenseNet121"]
