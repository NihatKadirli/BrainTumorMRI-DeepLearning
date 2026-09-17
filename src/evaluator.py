"""
evaluator.py
------------
Model performans degerlendirme modulu.

Hesaplanan metrikler:
- Accuracy, Precision, Recall, F1-score (macro & weighted)
- Confusion matrix
- ROC-AUC (one-vs-rest)
- Classification report (per-class)
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from typing import Dict, Any

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)
#model degerlendirme
from . import config


class ModelEvaluator:
    """Test seti uzerinde kapsamli model degerlendirmesi."""

    def __init__(self, model: tf.keras.Model, model_name: str) -> None:
        self.model = model
        self.model_name = model_name

    def evaluate(self, test_gen) -> Dict[str, Any]:
        """Test generator uzerinde tum metrikleri hesaplar."""
        # Tahminler
        test_gen.reset()
        y_pred_probs = self.model.predict(test_gen, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = test_gen.classes[: len(y_pred)]

        # Metrikler
        acc = accuracy_score(y_true, y_pred)
        prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec_w = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec_w = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        try:
            auc = roc_auc_score(
                tf.keras.utils.to_categorical(y_true, num_classes=config.NUM_CLASSES),
                y_pred_probs,
                average="macro",
                multi_class="ovr",
            )
        except ValueError:
            auc = float("nan")

        cm = confusion_matrix(y_true, y_pred)
        clf_report = classification_report(
            y_true, y_pred, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0,
        )

        results = {
            "model_name": self.model_name,
            "accuracy": float(acc),
            "precision_macro": float(prec_macro),
            "recall_macro": float(rec_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(prec_w),
            "recall_weighted": float(rec_w),
            "f1_weighted": float(f1_w),
            "auc_macro": float(auc),
            "confusion_matrix": cm.tolist(),
            "classification_report": clf_report,
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "y_pred_probs": y_pred_probs.tolist(),
        }

        self._save(results)
        return results

    def _save(self, results: Dict[str, Any]) -> None:
        out = config.RESULTS_DIR / f"{self.model_name}_evaluation.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[OK] {self.model_name} degerlendirme sonuclari kaydedildi: {out}")


def build_comparison_table(results_list) -> pd.DataFrame:
    """Tum modellerin karsilastirma tablosunu olusturur."""
    rows = []
    for r in results_list:
        rows.append({
            "Model": r["model_name"],
            "Accuracy": r["accuracy"],
            "Precision (macro)": r["precision_macro"],
            "Recall (macro)": r["recall_macro"],
            "F1 (macro)": r["f1_macro"],
            "AUC (macro)": r["auc_macro"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "comparison_table.csv", index=False)
    return df
