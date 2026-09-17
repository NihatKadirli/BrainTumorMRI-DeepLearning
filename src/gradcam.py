"""
gradcam.py
----------
Aciklanabilir Yapay Zeka (XAI): Grad-CAM (Gradient-weighted Class Activation Mapping)

Selvaraju et al. 2017 - "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization" makalesine dayali implementasyon.

Grad-CAM, son konvolusyon katmanindaki feature map'lerin, hedef sinifin skoruna
gore gradyanlarini kullanarak agirlikli bir heatmap uretir. Bu heatmap modelin
karar verirken hangi piksellere odaklandigini gosterir.

T1 tipi MRI goruntulerinde tumor bolgesinin modelin dikkati ile ortusmesi,
modelin klinik olarak anlamli ozellikleri yakaladiginin gostergesidir.
"""

from __future__ import annotations
import numpy as np
import tensorflow as tf
import cv2
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt

from . import config


class GradCAM:
    """Grad-CAM heatmap uretici."""

    def __init__(self, model: tf.keras.Model, last_conv_layer_name: str = None) -> None:
        self.model = model
        if last_conv_layer_name is None:
            last_conv_layer_name = self._find_last_conv_layer()
        self.last_conv_layer_name = last_conv_layer_name

        self.grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output,
            ],
        )

    def _find_last_conv_layer(self) -> str:
        """Modeldeki son Conv2D katmanini bulur."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("Modelde Conv2D katmani bulunamadi.")

    def compute_heatmap(
        self,
        image: np.ndarray,
        class_idx: int = None,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, int, float]:
        """Tek bir gorsel icin Grad-CAM heatmap'ini hesaplar."""
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image)
            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]))
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]

        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + eps)

        confidence = float(predictions[0][class_idx].numpy())
        return heatmap, class_idx, confidence

    def overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Heatmap'i orijinal gorsel uzerine bindirir."""
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored = cv2.applyColorMap(heatmap_uint8, colormap)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        img_uint8 = np.uint8(255 * image) if image.max() <= 1.0 else image.astype(np.uint8)
        overlayed = cv2.addWeighted(img_uint8, 1 - alpha, colored, alpha, 0)
        return overlayed

    def visualize_batch(
        self,
        images: np.ndarray,
        true_labels: List[int],
        save_path: Path,
    ) -> None:
        """Birden fazla gorsel icin Grad-CAM visualization kaydeder."""
        n = len(images)
        fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        for i, img in enumerate(images):
            heatmap, pred_idx, conf = self.compute_heatmap(img)
            overlayed = self.overlay(img, heatmap)

            axes[i, 0].imshow(img)
            axes[i, 0].set_title(f"Orijinal\nGercek: {config.CLASS_NAMES[true_labels[i]]}")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(heatmap, cmap="jet")
            axes[i, 1].set_title("Grad-CAM Heatmap")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(overlayed)
            axes[i, 2].set_title(
                f"Tahmin: {config.CLASS_NAMES[pred_idx]}\nGuven: {conf:.2%}"
            )
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Grad-CAM gorselleri kaydedildi: {save_path}")
