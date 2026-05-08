"""
Train skin condition classifier from images (ingredients + skin disease images).
Use for recommending SkinMe products by predicted condition.

Data options (single storage root):
  1) Folder structure:  data/skin_condition_images/<condition>/<image>.jpg
  2) CSV:               data/skin_condition_images/labels.csv with columns image_name, condition
                        and images in data/skin_condition_images/

Usage:
  python -m scripts.train_skin_condition
  python -m scripts.train_skin_condition --images data/skin_condition_images --csv data/skin_condition_images/labels.csv
  python -m scripts.train_skin_condition --epochs 10 --batch-size 8
"""
import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from skin_assistant.models.skin_condition_trainer import train_skin_condition_classifier
from skin_assistant.config import get_settings


def main():
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Train 2-model skin condition ensemble from images")
    parser.add_argument("--images", type=Path, default=None, help="Images dir (default: data/skin_condition_images)")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV with image_name, condition")
    parser.add_argument("--output", type=Path, default=None, help="Output dir for model (default: models/artifacts)")
    parser.add_argument("--image-col", type=str, default="image_name", help="CSV column for image filename")
    parser.add_argument("--condition-col", type=str, default="condition", help="CSV column for condition label")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-accuracy", type=float, default=0.95, help="Desired minimum validation accuracy")
    args = parser.parse_args()

    result = train_skin_condition_classifier(
        images_dir=args.images or settings.skin_disease_images_dir,
        labels_csv=args.csv or (settings.skin_disease_labels_path if settings.skin_disease_labels_path.exists() else None),
        output_dir=args.output or settings.models_dir,
        image_col=args.image_col,
        condition_col=args.condition_col,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_accuracy=args.target_accuracy,
    )
    print(json.dumps(result, indent=2))
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
