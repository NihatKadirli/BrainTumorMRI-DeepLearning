"""
custom_cnn.py
-------------
Bu proje icin tasarlanan OZEL CNN mimarisi (BrainNet-v1).

Tasarim gerekceleri:
- 4 konvolusyonel blok (giderek artan filtre sayisi): 32 -> 64 -> 128 -> 256
- Her blokta BatchNormalization: egitim stabilitesi
- SpatialDropout2D: overfitting'e karsi feature map seviyesinde duzenlileme
- Global Average Pooling: parametre sayisini dusurur, overfitting'i azaltir
- Dense + Dropout blok: siniflandirma katmani
- Toplam parametre sayisi ~2.3M (modern mimarilere gore cok daha kucuk)
"""

from tensorflow.keras import layers, models, regularizers
from .. import config


def _conv_block(x, filters: int, block_id: int, dropout: float = 0.1):
    """Tek bir konvolusyonel blok: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Pool -> Dropout"""
    x = layers.Conv2D(
        filters, (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"conv{block_id}_1",
    )(x)
    x = layers.BatchNormalization(name=f"bn{block_id}_1")(x)
    x = layers.Activation("relu", name=f"relu{block_id}_1")(x)

    x = layers.Conv2D(
        filters, (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-4),
        name=f"conv{block_id}_2",
    )(x)
    x = layers.BatchNormalization(name=f"bn{block_id}_2")(x)
    x = layers.Activation("relu", name=f"relu{block_id}_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name=f"pool{block_id}")(x)
    x = layers.SpatialDropout2D(dropout, name=f"sdrop{block_id}")(x)
    return x

#model tanimi
def build_custom_cnn(
    input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
    num_classes: int = config.NUM_CLASSES,
) -> models.Model:
    """
    Kendi CNN mimarimiz: BrainNet-v1.

    Parameters
    ----------
    input_shape : tuple
        Giris gorsel boyutu (H, W, C)
    num_classes : int
        Sinif sayisi

    Returns
    -------
    tf.keras.Model
        Derlenmeye hazir CNN modeli
    """
    inputs = layers.Input(shape=input_shape, name="input_layer")

    x = _conv_block(inputs, 32, 1, dropout=0.10)
    x = _conv_block(x, 64, 2, dropout=0.15)
    x = _conv_block(x, 128, 3, dropout=0.20)
    x = _conv_block(x, 256, 4, dropout=0.25)

    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4), name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(0.5, name="dropout_final")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs, outputs, name="BrainNet_v1")
    return model


if __name__ == "__main__":
    m = build_custom_cnn()
    m.summary()
