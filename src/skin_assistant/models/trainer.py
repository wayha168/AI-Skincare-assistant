"""
Training pipeline for intent classification model (scikit-learn).
Saves vectorizer + classifier to models/artifacts for use by the API.
"""
from collections import Counter
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from skin_assistant.config import get_settings
from skin_assistant.models.intent_labels import INTENT_LABELS, INTENT_LABEL_TO_ID


# Example training data: (text, intent). Expand with real user queries for better accuracy.
DEFAULT_TRAINING_DATA = [
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey there", "greeting"),
    ("what is niacinamide", "ingredient_info"),
    ("tell me about hyaluronic acid", "ingredient_info"),
    ("what does retinol do", "ingredient_info"),
    ("explain vitamin c", "ingredient_info"),
    ("products with niacinamide", "products_with_ingredient"),
    ("products containing hyaluronic acid", "products_with_ingredient"),
    ("recommend products with retinol", "products_with_ingredient"),
    ("recommend for dry skin", "product_recommendation"),
    ("I have acne", "product_recommendation"),
    ("products for sensitive skin", "product_recommendation"),
    ("best moisturiser for oily skin", "product_recommendation"),
    ("ingredients for hydration", "general_ingredient_search"),
    ("search ceramide", "general_ingredient_search"),
    ("something random", "other"),
]


def build_pipeline() -> Pipeline:
    """Build TF-IDF + LogisticRegression pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=500, C=0.5, random_state=42)),
        ]
    )


def _find_intent_training_csv(csv_path: Optional[Path] = None) -> Optional[Path]:
    settings = get_settings()
    if csv_path and csv_path.exists():
        return Path(csv_path)
    primary = settings.data_dir / "intent_training.csv"
    if primary.exists():
        return primary
    example = settings.data_dir / "intent_training.csv.example"
    if example.exists():
        return example
    return None


def _augment_intent_examples_from_ingredients() -> list[tuple[str, str]]:
    settings = get_settings()
    path = settings.ingredients_path
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "name" not in df.columns:
        return []
    examples = []
    for ingredient in df["name"].dropna().astype(str).str.strip().unique().tolist()[:200]:
        if not ingredient:
            continue
        examples.append((f"what is {ingredient}", "ingredient_info"))
        examples.append((f"tell me about {ingredient}", "ingredient_info"))
    return examples


def load_training_data(csv_path: Optional[Path] = None) -> tuple[list[str], list[int]]:
    """
    Load training data. If csv_path exists, expect columns 'text' and 'intent'.
    Otherwise use a data/ CSV fallback or built-in defaults.
    """
    source_path = _find_intent_training_csv(csv_path)
    if source_path:
        df = pd.read_csv(source_path)
        if "text" in df.columns and "intent" in df.columns:
            texts = df["text"].astype(str).tolist()
            intents = df["intent"].map(INTENT_LABEL_TO_ID).tolist()
        else:
            texts = []
            intents = []
    else:
        texts = [t for t, _ in DEFAULT_TRAINING_DATA]
        intents = [INTENT_LABEL_TO_ID[l] for _, l in DEFAULT_TRAINING_DATA]
    if source_path and source_path.name.endswith(".example"):
        extra = _augment_intent_examples_from_ingredients()
        for text, label in extra:
            texts.append(text)
            intents.append(INTENT_LABEL_TO_ID[label])
    return texts, intents


def train(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train intent classifier and save pipeline to output_dir.
    Returns metrics dict (accuracy, etc.).
    """
    settings = get_settings()
    output_dir = output_dir or settings.models_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    texts, y = load_training_data(data_path)
    if not texts:
        raise ValueError("No training data. Provide a CSV with 'text' and 'intent' or use defaults.")

    # Stratify only when every class has at least 2 samples (required by train_test_split)
    counts = Counter(y)
    use_stratify = all(c >= 2 for c in counts.values())
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if use_stratify else None,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    out_path = output_dir / "intent_model.joblib"
    joblib.dump(pipeline, out_path)

    return {
        "accuracy": float(score),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "model_path": str(out_path),
    }
