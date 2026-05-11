"""Train and infer skin condition models from uploaded image data."""
from __future__ import annotations

import csv
import json
import io
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

from skin_assistant.config import get_settings


def _collect_image_labels_from_folders(images_dir: Path) -> Tuple[List[str], List[str]]:
    """Scan images_dir for subdirs; each subdir name = condition, files = images. Returns (paths, labels)."""
    paths, labels = [], []
    if not images_dir.exists():
        return paths, labels
    for subdir in sorted(images_dir.iterdir()):
        if not subdir.is_dir():
            continue
        condition = subdir.name
        for f in subdir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                paths.append(str(f))
                labels.append(condition)
    return paths, labels


def _collect_image_labels_from_csv(
    csv_path: Path, images_dir: Path, image_col: str = "image_name", condition_col: str = "condition"
) -> Tuple[List[str], List[str]]:
    """CSV with image_col and condition_col; images live in images_dir (filename = image_name or path)."""
    paths, labels = [], []
    if not csv_path.exists() or not images_dir.exists():
        return paths, labels
    df = pd.read_csv(csv_path)
    if image_col not in df.columns or condition_col not in df.columns:
        return paths, labels
    for _, row in df.iterrows():
        name = str(row[image_col]).strip()
        cond = str(row[condition_col]).strip()
        if not name or not cond:
            continue
        # Support full path or just filename
        p = images_dir / name
        if not p.exists():
            p = images_dir / (name if not Path(name).suffix else name + ".jpg")
        if p.exists():
            paths.append(str(p))
            labels.append(cond)
    return paths, labels


def save_uploaded_skin_image_for_training(
    image_bytes: bytes,
    original_filename: str = "",
    user_message: str = "",
    session_id: Optional[str] = None,
    condition_label: Optional[str] = None,
) -> Optional[Path]:
    """
    Save user-uploaded image in one central training folder.

    - If condition_label is provided, image is saved under <data>/<condition_label>/...
    - Else image is saved under <data>/unlabeled/...
    - Metadata is appended to uploads_metadata.csv for later labeling/training.
    """
    settings = get_settings()
    base_dir = settings.skin_condition_data_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    safe_label = (condition_label or "").strip().lower().replace(" ", "_")
    target_label = safe_label if safe_label else "unlabeled"
    target_dir = base_dir / target_label
    target_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(original_filename or "").suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        ext = ".jpg"
    filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:10]}{ext}"
    out_path = target_dir / filename

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.save(out_path)
    except Exception:
        return None

    metadata_path = settings.skin_uploads_metadata_path
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not metadata_path.exists()
    with metadata_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "timestamp_utc",
                    "relative_path",
                    "condition_label",
                    "session_id",
                    "original_filename",
                    "user_message",
                ]
            )
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                str(out_path.relative_to(base_dir)),
                condition_label or "",
                session_id or "",
                original_filename or "",
                user_message or "",
            ]
        )

    return out_path


def _build_model(model_name: str, num_classes: int):
    """Create torchvision model and replace classifier head."""
    from torchvision import models
    import torch

    if model_name == "resnet18":
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            try:
                model = models.resnet18(pretrained=True)
            except Exception:
                model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        try:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:
            try:
                model = models.efficientnet_b0(pretrained=True)
            except Exception:
                model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model_name: {model_name}")


def train_skin_condition_classifier(
    images_dir: Optional[Path] = None,
    labels_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    image_col: str = "image_name",
    condition_col: str = "condition",
    epochs: int = 8,
    batch_size: int = 16,
    image_size: int = 224,
    target_accuracy: float = 0.95,
) -> dict:
    """
    Train 2 image models (ResNet18 + EfficientNet-B0) for skin conditions.
    Saves checkpoints to output_dir/skin_condition_model_<model>.pt.
    If labels_csv is set, uses CSV (image_name, condition); else uses folder structure images_dir/<condition>/*.jpg.
    """
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader, random_split
        from torchvision import transforms
    except ImportError:
        return {"error": "Install torch, torchvision, Pillow: pip install torch torchvision Pillow"}

    settings = get_settings()
    images_dir = images_dir or settings.skin_disease_images_dir
    output_dir = output_dir or settings.models_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if labels_csv and Path(labels_csv).exists():
        image_paths, image_labels = _collect_image_labels_from_csv(
            Path(labels_csv), images_dir, image_col=image_col, condition_col=condition_col
        )
    else:
        image_paths, image_labels = _collect_image_labels_from_folders(images_dir)

    if len(image_paths) < 2 or len(set(image_labels)) < 2:
        return {
            "error": "Need at least 2 images and 2 classes. Use folder structure data/skin_condition_images/<condition>/*.jpg or CSV with image_name, condition.",
            "hint_images_dir": str(images_dir),
            "hint_csv": str(labels_csv or settings.skin_disease_labels_path),
        }

    label_to_id = {v: i for i, v in enumerate(sorted(set(image_labels)))}
    id_to_label = {i: v for v, i in label_to_id.items()}
    num_classes = len(label_to_id)
    labels_ids = [label_to_id[l] for l in image_labels]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class SkinConditionDataset(Dataset):
        def __init__(self, paths, label_ids, transform_fn):
            self.paths = paths
            self.label_ids = label_ids
            self.transform_fn = transform_fn

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            try:
                img = Image.open(self.paths[i]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (image_size, image_size), (128, 128, 128))
            return self.transform_fn(img), self.label_ids[i]

    all_indices = list(range(len(image_paths)))
    val_size = max(1, int(0.2 * len(all_indices)))
    split = random_split(
        all_indices,
        [len(all_indices) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_indices = [all_indices[i] for i in split[0].indices]
    val_indices = [all_indices[i] for i in split[1].indices]

    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels_ids[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels_ids[i] for i in val_indices]

    train_ds = SkinConditionDataset(train_paths, train_labels, train_transform)
    val_ds = SkinConditionDataset(val_paths, val_labels, eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.CrossEntropyLoss()

    def train_one_model(model_name: str) -> Tuple[Path, float, List[Dict[str, float]]]:
        model = _build_model(model_name=model_name, num_classes=num_classes).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

        best_acc = 0.0
        best_state = None
        history: List[Dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            model.train()
            train_correct, train_total = 0, 0
            for xs, ys in train_loader:
                xs, ys = xs.to(device), ys.to(device)
                opt.zero_grad()
                out = model(xs)
                loss = criterion(out, ys)
                loss.backward()
                opt.step()
                pred = out.argmax(dim=1)
                train_correct += int((pred == ys).sum().item())
                train_total += int(ys.size(0))

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xs, ys in val_loader:
                    xs, ys = xs.to(device), ys.to(device)
                    out = model(xs)
                    pred = out.argmax(dim=1)
                    correct += int((pred == ys).sum().item())
                    total += int(ys.size(0))
            train_acc = float(train_correct / train_total) if train_total else 0.0
            acc = float(correct / total) if total else 0.0
            history.append(
                {
                    "epoch": float(epoch),
                    "train_accuracy": train_acc,
                    "validation_accuracy": acc,
                }
            )
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        out_path = output_dir / f"skin_condition_model_{model_name}.pt"
        torch.save(
            {
                "model_name": model_name,
                "model_state": best_state or model.state_dict(),
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "num_classes": num_classes,
                "validation_accuracy": best_acc,
                "target_accuracy": target_accuracy,
                "history": history,
            },
            out_path,
        )
        return out_path, best_acc, history

    model_results: Dict[str, Dict[str, object]] = {}
    history_by_model: Dict[str, List[Dict[str, float]]] = {}
    ensemble_models = ["resnet18", "efficientnet_b0"]
    for model_name in ensemble_models:
        out_path, acc, history = train_one_model(model_name)
        model_results[model_name] = {"model_path": str(out_path), "validation_accuracy": acc}
        history_by_model[model_name] = history

    history_csv_path = output_dir / "skin_condition_training_history.csv"
    with history_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["model_name", "epoch", "train_accuracy", "validation_accuracy"])
        for model_name, history in history_by_model.items():
            for item in history:
                writer.writerow(
                    [
                        model_name,
                        int(item["epoch"]),
                        item["train_accuracy"],
                        item["validation_accuracy"],
                    ]
                )

    history_json_path = output_dir / "skin_condition_training_history.json"
    history_payload = {
        "epochs": epochs,
        "samples": len(image_paths),
        "num_classes": num_classes,
        "classes": list(id_to_label.values()),
        "models": history_by_model,
    }
    history_json_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")

    avg_acc = float(sum(float(v["validation_accuracy"]) for v in model_results.values()) / len(model_results))
    meets_target = avg_acc >= target_accuracy

    return {
        "epochs": epochs,
        "samples": len(image_paths),
        "num_classes": num_classes,
        "classes": list(id_to_label.values()),
        "target_accuracy": target_accuracy,
        "ensemble_validation_accuracy": avg_acc,
        "meets_target_accuracy": meets_target,
        "models": model_results,
        "history_csv": str(history_csv_path),
        "history_json": str(history_json_path),
        "note": (
            "Current validation accuracy is below target. Add more balanced, clean labeled data "
            "to reach >=95% consistently."
            if not meets_target
            else "Target accuracy reached on validation split."
        ),
    }


def predict_skin_condition_from_image(image_input, model_path: Optional[Path] = None, image_size: int = 224):
    """
    Predict skin condition from a PIL Image or path to an image file.
    Returns (condition_label, confidence) or (None, 0) if model missing or inference fails.
    """
    try:
        import torch
        from torchvision import transforms, models
        from PIL import Image
    except ImportError:
        return None, 0.0

    settings = get_settings()
    if model_path:
        candidate_paths = [Path(model_path)]
    else:
        candidate_paths = [
            settings.models_dir / "skin_condition_model_resnet18.pt",
            settings.models_dir / "skin_condition_model_efficientnet_b0.pt",
        ]
    available_paths = [p for p in candidate_paths if p.exists()]
    if not available_paths:
        return None, 0.0

    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB") if hasattr(image_input, "convert") else Image.fromarray(image_input).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = transform(img).unsqueeze(0)
    total_probs = None
    id_to_label = {}

    for path in available_paths:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            num_classes = checkpoint.get("num_classes", 0)
            ckpt_labels = checkpoint.get("id_to_label", {})
            model_name = checkpoint.get("model_name") or (
                "efficientnet_b0" if "efficientnet" in path.name else "resnet18"
            )
            model = _build_model(model_name=model_name, num_classes=num_classes)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            with torch.no_grad():
                out = model(x)
                probs = torch.softmax(out, dim=1)
            total_probs = probs if total_probs is None else total_probs + probs
            if ckpt_labels:
                id_to_label = ckpt_labels
        except Exception:
            continue

    if total_probs is None:
        return None, 0.0

    avg_probs = total_probs / max(1, len(available_paths))
    conf, idx = avg_probs[0].max(0).item(), avg_probs[0].argmax(0).item()
    label = id_to_label.get(int(idx), id_to_label.get(idx, "unknown"))
    return label, float(conf)
