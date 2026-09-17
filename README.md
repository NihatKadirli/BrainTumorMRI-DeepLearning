# 🧠 Brain Tumor MRI Classification with Deep Learning

A comparative deep learning project for classifying brain MRI images
into four categories:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The project compares four deep learning architectures:

- BrainNet-v1 (Custom CNN)
- ResNet50
- EfficientNetB0
- DenseNet121

Explainability is provided using Grad-CAM to visualize the image
regions influencing model predictions.

## 🏆 Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| BrainNet-v1 | 95.04% | 94.84% | 95.07% | 94.94% | 96.03% |
| ResNet50 | 97.10% | 97.01% | 97.07% | 97.04% | 97.68% |
| EfficientNetB0 | 98.02% | 97.94% | 98.03% | 97.98% | 98.41% |
| DenseNet121 | 98.86% | 98.82% | 98.90% | 98.86% | 99.08% |

## 🔍 Explainable AI

Grad-CAM is used to visualize which MRI regions contribute most
to the model's predictions.

## ⚠️ Disclaimer

This project was developed for educational and research purposes.
It is not a medical diagnostic tool and should not be used for
clinical decision-making.
