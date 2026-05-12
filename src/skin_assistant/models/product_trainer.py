"""
Train models from product data (CSV) and optionally from product images.
- Product text model: predict productType/category from name + description (sklearn).
- Image model: predict productType from image using PyTorch/torchvision (optional).
"""
from pathlib import Path
from typing import Optional

import pandas as pd

from skin_assistant.config import get_settings


def load_skinme_products(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load SkinMe products CSV from available data files."""
    settings = get_settings()
    candidates = []
    if csv_path:
        candidates.append(Path(csv_path))
    candidates.extend([settings.skinme_products_path, settings.products_path])
    for path in candidates:
        if path and path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                df.attrs["source_csv_path"] = str(path)
                return df
    return pd.DataFrame()


def _resolve_text_columns(df: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, str]:
    if "name" not in df.columns and "product_name" in df.columns:
        df = df.copy()
        df["name"] = df["product_name"].astype(str)
    if label_column not in df.columns:
        if label_column == "productType" and "product_type" in df.columns:
            label_column = "product_type"
        elif label_column == "category_name" and "category_name" in df.columns:
            label_column = "category_name"
        elif "product_type" in df.columns:
            label_column = "product_type"
        elif "category_name" in df.columns:
            label_column = "category_name"
    return df, label_column


def train_product_type_from_text(
    products_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    label_column: str = "productType",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train a classifier to predict productType (or category_name) from product name + description.
    Uses TF-IDF + LogisticRegression. Saves to output_dir/product_type_model.joblib.
    """
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    settings = get_settings()
    output_dir = output_dir or settings.models_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_skinme_products(products_csv)
    if df.empty:
        return {"error": "No product CSV found. Run sync_products first or add a products CSV to data/."}
    df, label_column = _resolve_text_columns(df, label_column)
    if "name" not in df.columns or label_column not in df.columns:
        return {"error": f"CSV must have a text column ('name' or 'product_name') and a label column like '{label_column}'."}
    # Build text: name + description
    df = df.dropna(subset=[label_column])
    df["_text"] = (df["name"].fillna("") + " " + df.get("description", pd.Series("")).fillna("")).str.strip()
    X = df["_text"].tolist()
    y = df[label_column].astype(str).str.strip().tolist()
    if len(set(y)) < 2:
        return {"error": "Need at least 2 distinct labels to train."}
    # Stratify only when every class has at least 2 samples (required by train_test_split)
    from collections import Counter
    min_count = min(Counter(y).values())
    use_stratify = min_count >= 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if use_stratify else None
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=500, C=0.5, random_state=random_state)),
    ])
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    out_path = output_dir / "product_type_model.joblib"
    joblib.dump(pipeline, out_path)
    return {
        "accuracy": float(score),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "label_column": label_column,
        "model_path": str(out_path),
    }


def train_image_classifier(
    products_csv: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    label_column: str = "productType",
    epochs: int = 5,
    batch_size: int = 8,
    image_size: int = 224,
) -> dict:
    """
    Train image classifier: labels from product CSV, images from product_images_dir.
    Expects filenames like {product_id}_{image_id}.jpg. Uses PyTorch + torchvision.
    """
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torchvision import transforms, models
        from PIL import Image
    except ImportError:
        return {"error": "Install torch, torchvision, Pillow: pip install torch torchvision Pillow"}

    settings = get_settings()
    csv_path = products_csv or settings.skinme_products_path
    images_dir = images_dir or settings.product_images_dir
    output_dir = output_dir or settings.models_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        return {"error": "Product CSV not found. Run sync_products first."}
    if not images_dir.exists():
        return {"error": "Product images dir not found. Run sync_products (with download)."}

    df = pd.read_csv(csv_path)
    if df.empty or "id" not in df.columns or label_column not in df.columns:
        return {"error": f"CSV must have 'id' and '{label_column}'."}
    df = df.dropna(subset=[label_column])
    labels = df[label_column].astype(str)
    label_to_id = {v: i for i, v in enumerate(sorted(labels.unique()))}
    id_to_label = {i: v for v, i in label_to_id.items()}

    # Find image files that match product ids
    image_paths = []
    image_labels = []
    for _, row in df.iterrows():
        pid = str(row["id"])
        lab = str(row[label_column])
        if lab not in label_to_id:
            continue
        for f in images_dir.glob(f"{pid}_*"):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".avif"):
                image_paths.append(str(f))
                image_labels.append(label_to_id[lab])
                break

    if len(image_paths) < 2 or len(set(image_labels)) < 2:
        return {"error": "Not enough images or labels. Need at least 2 classes and 2 images."}

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class ProductImageDataset(Dataset):
        def __init__(self, paths, labels, transform_fn):
            self.paths = paths
            self.labels = labels
            self.transform_fn = transform_fn

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            try:
                img = Image.open(self.paths[i]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (image_size, image_size), (128, 128, 128))
            return self.transform_fn(img), self.labels[i]

    dataset = ProductImageDataset(image_paths, image_labels, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(label_to_id)
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except AttributeError:
        model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xs, ys in loader:
            xs, ys = xs.to(device), ys.to(device)
            opt.zero_grad()
            out = model(xs)
            loss = criterion(out, ys)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        # no validation for minimal script

    out_path = output_dir / "product_image_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "num_classes": num_classes,
    }, out_path)

    return {
        "epochs": epochs,
        "samples": len(image_paths),
        "num_classes": num_classes,
        "model_path": str(out_path),
    }
