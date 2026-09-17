"""Model mimarileri paketi."""
from .custom_cnn import build_custom_cnn
from .transfer_models import build_resnet50, build_efficientnet_b0, build_densenet121

MODEL_BUILDERS = {
    "CustomCNN": build_custom_cnn,
    "ResNet50": build_resnet50,
    "EfficientNetB0": build_efficientnet_b0,
    "DenseNet121": build_densenet121,
}

__all__ = [
    "build_custom_cnn",
    "build_resnet50",
    "build_efficientnet_b0",
    "build_densenet121",
    "MODEL_BUILDERS",
]
