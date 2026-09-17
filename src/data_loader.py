"""
data_loader.py
--------------
Veri seti yukleme, on-isleme ve augmentation islemleri.

Bu modul Brain Tumor MRI Dataset'ini tf.data pipeline'i ile yukler.
Uretim seviyesinde veri hazirlik sinifi (class-based) olarak tasarlanmistir.
"""

from __future__ import annotations
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
from typing import Tuple, Dict
import numpy as np

from . import config
#veri yukleme

class BrainTumorDataLoader:
    """Brain Tumor MRI veri seti icin kapsulleyici sinif."""

    def __init__(
        self,
        train_dir: Path = config.TRAIN_DIR,
        test_dir: Path = config.TEST_DIR,
        img_size: int = config.IMG_SIZE,
        batch_size: int = config.BATCH_SIZE,
        validation_split: float = config.VALIDATION_SPLIT,
        seed: int = config.RANDOM_SEED,
    ) -> None:
        self.train_dir = Path(train_dir)
        self.test_dir = Path(test_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.seed = seed

        self._validate_paths()

    def _validate_paths(self) -> None:
        """Veri seti yollarinin varligini dogrular."""
        if not self.train_dir.exists():
            raise FileNotFoundError(
                f"Training klasoru bulunamadi: {self.train_dir}\n"
                f"Lutfen veri setini {config.DATASET_URL} adresinden indirin."
            )
        if not self.test_dir.exists():
            raise FileNotFoundError(f"Testing klasoru bulunamadi: {self.test_dir}")

    def _build_train_augmenter(self) -> ImageDataGenerator:
        """Egitim icin augmentation pipeline'i."""
        return ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=config.AUG_ROTATION,
            width_shift_range=config.AUG_WIDTH_SHIFT,
            height_shift_range=config.AUG_HEIGHT_SHIFT,
            zoom_range=config.AUG_ZOOM,
            horizontal_flip=config.AUG_HORIZONTAL_FLIP,
            fill_mode="nearest",
            validation_split=self.validation_split,
        )

    def _build_eval_augmenter(self) -> ImageDataGenerator:
        """Validation/Test icin sadece normalizasyon."""
        return ImageDataGenerator(rescale=1.0 / 255.0)

    def get_generators(self) -> Tuple:
        """Train, validation ve test generator'larini dondurur."""
        train_aug = self._build_train_augmenter()
        eval_aug = self._build_eval_augmenter()

        train_gen = train_aug.flow_from_directory(
            self.train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode="categorical",
            classes=config.CLASS_NAMES,
            subset="training",
            shuffle=True,
            seed=self.seed,
        )

        val_gen = train_aug.flow_from_directory(
            self.train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode="categorical",
            classes=config.CLASS_NAMES,
            subset="validation",
            shuffle=False,
            seed=self.seed,
        )

        test_gen = eval_aug.flow_from_directory(
            self.test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode="categorical",
            classes=config.CLASS_NAMES,
            shuffle=False,
        )

        return train_gen, val_gen, test_gen

    def get_class_distribution(self) -> Dict[str, Dict[str, int]]:
        """Egitim ve test setleri icin sinif dagilimini dondurur."""
        distribution = {"train": {}, "test": {}}
        for split, path in [("train", self.train_dir), ("test", self.test_dir)]:
            for cls in config.CLASS_NAMES:
                cls_dir = path / cls
                count = len(list(cls_dir.glob("*.*"))) if cls_dir.exists() else 0
                distribution[split][cls] = count
        return distribution

    def compute_class_weights(self, train_gen) -> Dict[int, float]:
        """Dengesiz sinif dagilimi icin agirlik hesaplar."""
        from sklearn.utils.class_weight import compute_class_weight

        y_train = train_gen.classes
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        return {int(c): float(w) for c, w in zip(classes, weights)}


if __name__ == "__main__":
    loader = BrainTumorDataLoader()
    print("Sinif dagilimi:", loader.get_class_distribution())
    train_gen, val_gen, test_gen = loader.get_generators()
    print(f"Egitim: {train_gen.samples}, Val: {val_gen.samples}, Test: {test_gen.samples}")
