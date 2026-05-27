"""Application settings and paths."""
import os
from pathlib import Path
from dotenv import load_dotenv


# Load .env from project root
root = Path(__file__).resolve().parents[3]  # config -> skin_assistant -> src -> project root
load_dotenv(root / '.env')


def get_settings():
    """Return settings object (paths + optional backend URL and MySQL for DB integration)."""
    # Spring (or other) backend URL for saving data to database; e.g. http://localhost:8080
    backend_url = os.environ.get("SPRING_BACKEND_URL") or os.environ.get("BACKEND_WEBHOOK_URL") or ""
    # Optional: MySQL (skinme_db) for product lookups when "use database" is enabled in chat
    mysql_host = os.environ.get("MYSQL_HOST", "").strip()
    mysql_port = int(os.environ.get("MYSQL_PORT", "3306"))
    mysql_user = os.environ.get("MYSQL_USER", "").strip()
    mysql_password = os.environ.get("MYSQL_PASSWORD", "").strip()
    mysql_database = os.environ.get("MYSQL_DATABASE", "").strip()
    mysql_products_table = os.environ.get("MYSQL_PRODUCTS_TABLE", "product").strip() or "product"
    use_mysql_db = bool(mysql_host and mysql_user and mysql_database)
    # Override with CHAT_USE_CSV_ONLY if set to true (case-insensitive)
    chat_use_csv_only = os.environ.get("CHAT_USE_CSV_ONLY", "").lower()
    if chat_use_csv_only == "true":
        use_mysql_db = False
    # Product JSON API (same data as storefront; default is proxied on skinme.store)
    skinme_api_url = (
        os.environ.get("SKINME_API_URL", "").strip()
        or "https://skinme.store/api/v1/products/all"
    )
    # Backend origin (API may return absolute URLs pointing here; we rewrite image URLs to the storefront)
    skinme_base_url = os.environ.get("SKINME_BACKEND_URL", "").strip() or "https://backend.skinme.store"
    # Public site where /uploads/... assets are served (shareable image links)
    skinme_frontend_base_url = (
        os.environ.get("SKINME_FRONTEND_URL", "").strip()
        or os.environ.get("FRONTEND_URL", "").strip()
        or "https://skinme.store"
    ).rstrip("/")
    return type("Settings", (), {
        "project_root": root,
        "data_dir": root / "data",
        "ingredients_path": root / "data" / "ingredientsList.csv",
        "ingredient_lookup_path": root / "data" / "ingredient_lookup.csv",
        "products_path": root / "data" / "skincare_products_clean.csv",
        "dataset_products_path": root / "data" / "products_ingredients_dataset.csv",
        "models_dir": root / "models" / "artifacts",
        # SkinMe API & product CSV (image URLs use skinme_frontend_base_url for /uploads/...)
        "skinme_api_url": skinme_api_url,
        "skinme_base_url": skinme_base_url.rstrip("/"),
        "skinme_frontend_base_url": skinme_frontend_base_url,
        "skinme_products_path": root / "data" / "skinme_products.csv",
        "product_images_dir": root / "data" / "product_images",
        # Unified skin condition training storage:
        # data/skin_condition_images/<condition>/<image>.jpg for labeled images
        # data/skin_condition_images/unlabeled/<image>.jpg for user uploads pending label
        "skin_condition_data_dir": root / "data" / "skin_condition_images",
        "skin_disease_images_dir": root / "data" / "skin_condition_images",
        "skin_disease_labels_path": root / "data" / "skin_condition_images" / "labels.csv",
        "skin_uploads_metadata_path": root / "data" / "skin_condition_images" / "uploads_metadata.csv",
        # Backend integration: Spring URL to forward save requests (persist to DB)
        "backend_url": backend_url.rstrip("/") if backend_url else "",
        # Optional MySQL (skinme_db) for chat "check with database" option
        "mysql_host": mysql_host,
        "mysql_port": mysql_port,
        "mysql_user": mysql_user,
        "mysql_password": mysql_password,
        "mysql_database": mysql_database,
        "mysql_products_table": mysql_products_table,
        "use_mysql_db": use_mysql_db,
    })()
