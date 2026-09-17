"""
transfer_models.py
------------------
Modern derin ogrenme mimarileri icin transfer learning wrapper'lari.

Kullanilan mimariler:
- ResNet50   : Residual baglantilarla derin ag (He et al., 2015) - 25.6M param
- EfficientNetB0 : Compound scaling ile verimli mimari (Tan & Le, 2019) - 5.3M param
- DenseNet121 : Dense connection ile feature reuse (Huang et al., 2017) - 8.1M param

Her model iki asamada egitilir:
  1) Feature extraction: Taban donduk, sadece head egitilir (5 epoch)
  2) Fine-tuning       : Tum ag dusuk LR ile optimize edilir
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0, DenseNet121
from .. import config


def _build_transfer_head(base_model, num_classes: int, dropout: float = 0.3) -> models.Model:
    """Transfer learning icin standart classification head."""
    base_model.trainable = False  # Feature extraction asamasi icin donduruluyor

    inputs = base_model.input
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout, name="dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="dense_head")(x)
    x = layers.Dropout(dropout, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return models.Model(inputs, outputs, name=base_model.name + "_transfer")


def build_resnet50(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    num_classes: int = config.NUM_CLASSES,
) -> models.Model:
    """ResNet50 tabanli transfer learning modeli."""
    base = ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    return _build_transfer_head(base, num_classes)


def build_efficientnet_b0(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    num_classes: int = config.NUM_CLASSES,
) -> models.Model:
    """EfficientNetB0 tabanli transfer learning modeli."""
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)
    return _build_transfer_head(base, num_classes)


def build_densenet121(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    num_classes: int = config.NUM_CLASSES,
) -> models.Model:
    """DenseNet121 tabanli transfer learning modeli."""
    base = DenseNet121(include_top=False, weights="imagenet", input_shape=input_shape)
    return _build_transfer_head(base, num_classes)


def unfreeze_for_fine_tuning(model: models.Model, unfreeze_fraction: float = 0.3) -> None:
    """
    Fine-tuning icin son katmanlari acar.

    Parameters
    ----------
    model : tf.keras.Model
        Donmus taban model iceren model.
    unfreeze_fraction : float
        Modelin sondan hangi oraninin acilacagini belirler (0.0-1.0).
        Sabit indeks yerine oran kullanmak, farkli derinlikteki modellerde
        tutarli davranis saglar.
    """
    total = len(model.layers)
    unfreeze_from = int(total * (1.0 - unfreeze_fraction))
    for layer in model.layers[unfreeze_from:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True


if __name__ == "__main__":
    for name, builder in [
        ("ResNet50", build_resnet50),
        ("EfficientNetB0", build_efficientnet_b0),
        ("DenseNet121", build_densenet121),
    ]:
        m = builder()
        print(f"{name}: {m.count_params():,} parametre")
