"""
main.py
-------
Proje ana giris noktasi. Tum pipeline'i cagirir:
    1) Veri yukleme
    2) 4 modelin egitimi (CustomCNN, ResNet50, EfficientNetB0, DenseNet121)
    3) Degerlendirme ve karsilastirma
    4) Grad-CAM (XAI) analizi
    5) Grafik ve tablolarin kaydi

Kullanim:
    python main.py --models all
    python main.py --models CustomCNN,ResNet50
    python main.py --skip-training  # Sadece degerlendirme yap
"""

import argparse
import json

import numpy as np
import tensorflow as tf

from src import config
from src.data_loader import BrainTumorDataLoader
from src.models import MODEL_BUILDERS
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator, build_comparison_table
from src.gradcam import GradCAM
from src.visualizer import (
    plot_training_curves, plot_confusion_matrix, plot_comparison_bar,
    plot_class_distribution, plot_all_training_curves,
)


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)

#model egitimi
def run_pipeline(models_to_train: list, skip_training: bool = False) -> None:
    set_seed()
    print("=" * 70)
    print("BEYIN TUMORU MRI SINIFLANDIRMA - ANA PIPELINE")
    print("=" * 70)

    # 1. VERI YUKLEME
    print("\n[1/5] Veri seti yukleniyor...")
    loader = BrainTumorDataLoader()
    distribution = loader.get_class_distribution()
    print(f"Sinif dagilimi: {json.dumps(distribution, indent=2)}")

    plot_class_distribution(distribution, config.RESULTS_DIR / "class_distribution.png")

    train_gen, val_gen, test_gen = loader.get_generators()
    class_weights = loader.compute_class_weights(train_gen)
    print(f"Class weights: {class_weights}")

    # 2. MODEL EGITIMLERI
    all_histories = {}
    all_results = []

    for model_name in models_to_train:
        print(f"\n[2/5] Model egitiliyor: {model_name}")
        builder = MODEL_BUILDERS[model_name]
        model = builder()
        trainer = ModelTrainer(model, model_name)

        if not skip_training:
            if model_name == "CustomCNN":
                history = trainer.train(train_gen, val_gen, class_weights)
            else:
                history = trainer.train_two_stage(train_gen, val_gen, class_weights)
        else:
            # Daha onceki history'i yukle
            hist_path = config.RESULTS_DIR / f"{model_name}_history.json"
            with open(hist_path, "r") as f:
                history = json.load(f)

        all_histories[model_name] = history

        # Egitim egrilerini ciz
        plot_training_curves(
            history, model_name,
            config.RESULTS_DIR / f"{model_name}_training_curve.png",
        )

        # 3. DEGERLENDIRME
        print(f"\n[3/5] Test seti degerlendirmesi: {model_name}")
        # En iyi checkpoint'i yukle
        ckpt = config.CHECKPOINT_DIR / f"{model_name}_best.keras"
        if ckpt.exists():
            model = tf.keras.models.load_model(ckpt)

        evaluator = ModelEvaluator(model, model_name)
        results = evaluator.evaluate(test_gen)
        all_results.append(results)

        # Confusion matrix
        plot_confusion_matrix(
            np.array(results["confusion_matrix"]),
            model_name,
            config.RESULTS_DIR / f"{model_name}_confusion_matrix.png",
        )

        # 4. GRAD-CAM (XAI)
        print(f"\n[4/5] Grad-CAM uretiliyor: {model_name}")
        try:
            # Test setinden her siniftan 1 ornek al — tum siniflar bulunana kadar batchleri tara
            test_gen.reset()
            seen = {}  # cls -> (img, label)
            for _ in range(len(test_gen)):
                x_batch, y_batch = next(test_gen)
                for i, y in enumerate(y_batch):
                    cls = int(np.argmax(y))
                    if cls not in seen:
                        seen[cls] = (x_batch[i], cls)
                if len(seen) == config.NUM_CLASSES:
                    break

            sample_imgs = np.array([seen[c][0] for c in sorted(seen)])
            sample_labels = [seen[c][1] for c in sorted(seen)]

            gradcam = GradCAM(model)
            gradcam.visualize_batch(
                sample_imgs, sample_labels,
                config.RESULTS_DIR / f"{model_name}_gradcam.png",
            )
        except Exception as e:
            print(f"[UYARI] {model_name} icin Grad-CAM uretilemedi: {e}")

    # 5. KARSILASTIRMA
    print("\n[5/5] Modeller karsilastiriliyor...")
    comparison_df = build_comparison_table(all_results)
    print(comparison_df.to_string(index=False))

    plot_comparison_bar(comparison_df, config.RESULTS_DIR / "model_comparison.png")
    plot_all_training_curves(all_histories, config.RESULTS_DIR / "all_training_curves.png")

    print("\n" + "=" * 70)
    print(f"TAMAMLANDI! Tum sonuclar: {config.RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beyin Tumoru MRI Siniflandirma")
    parser.add_argument(
        "--models", default="all",
        help="Egitilecek modeller: 'all' veya virgulle ayrilmis isimler",
    )
    parser.add_argument(
        "--skip-training", action="store_true",
        help="Egitimi atla, sadece degerlendirme yap (onceden egitilmis modelle)",
    )
    args = parser.parse_args()

    if args.models == "all":
        models_list = config.MODEL_NAMES
    else:
        models_list = [m.strip() for m in args.models.split(",")]

    run_pipeline(models_list, skip_training=args.skip_training)
