"""
trainer.py
----------
Model egitim pipeline'i: Callback'ler, iki-asamali egitim, history kaydi.
"""

from __future__ import annotations
import json
from typing import Dict, Any, Optional

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger,
)
from tensorflow.keras.optimizers import Adam

from . import config
from .models.transfer_models import unfreeze_for_fine_tuning


class ModelTrainer:
    """Tek model icin egitim yoneticisi."""
#model egitimi
    def __init__(self, model: tf.keras.Model, model_name: str) -> None:
        self.model = model
        self.model_name = model_name
        self.history: Optional[Dict[str, Any]] = None
        self.ckpt_path = config.CHECKPOINT_DIR / f"{model_name}_best.keras"
        self.log_path = config.RESULTS_DIR / f"{model_name}_training_log.csv"

    def _build_callbacks(self) -> list:
        return [
            EarlyStopping(
                monitor="val_loss",
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=config.REDUCE_LR_FACTOR,
                patience=config.REDUCE_LR_PATIENCE,
                min_lr=config.MIN_LR,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(self.ckpt_path),
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
            CSVLogger(str(self.log_path), append=False),
        ]

    def compile_model(self, lr: float = config.INITIAL_LR) -> None:
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

    def train(self, train_gen, val_gen, class_weights: Dict[int, float] = None) -> Dict:
        """Tek-asamali egitim (CustomCNN icin)."""
        self.compile_model(lr=config.INITIAL_LR)
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.EPOCHS,
            callbacks=self._build_callbacks(),
            class_weight=class_weights,
            verbose=1,
        )
        self.history = history.history
        self._save_history()
        return self.history

    def train_two_stage(
        self,
        train_gen,
        val_gen,
        class_weights: Dict[int, float] = None,
        warmup_epochs: int = 5,
    ) -> Dict:
        """
        Iki-asamali egitim (transfer modeller icin):
        1) Feature extraction (warmup_epochs)
        2) Fine-tuning (kalan epoch'lar, dusuk LR)
        """
        # Stage 1: Feature extraction
        print(f"\n[Stage 1] Feature extraction - {self.model_name}")
        self.compile_model(lr=config.INITIAL_LR)
        h1 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=warmup_epochs,
            callbacks=[CSVLogger(str(self.log_path), append=False)],
            class_weight=class_weights,
            verbose=1,
        )

        # Stage 2: Fine-tuning
        print(f"\n[Stage 2] Fine-tuning - {self.model_name}")
        unfreeze_for_fine_tuning(self.model)
        self.compile_model(lr=config.FINE_TUNE_LR)
        stage2_callbacks = self._build_callbacks()
        # Overwrite the CSVLogger in callbacks with append=True so Stage 1 rows are kept
        stage2_callbacks = [
            cb if not isinstance(cb, CSVLogger)
            else CSVLogger(str(self.log_path), append=True)
            for cb in stage2_callbacks
        ]
        h2 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.EPOCHS,
            initial_epoch=warmup_epochs,
            callbacks=stage2_callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        # Merge histories — union of both key sets so h2-only keys (e.g. lr) are kept
        merged = {}
        for k in h1.history.keys() | h2.history.keys():
            merged[k] = list(h1.history.get(k, [])) + list(h2.history.get(k, []))
        self.history = merged
        self._save_history()
        return self.history

    def _save_history(self) -> None:
        out = config.RESULTS_DIR / f"{self.model_name}_history.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in self.history.items()},
                      f, indent=2)
