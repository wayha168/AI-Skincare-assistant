"""
Sync SkinMe products: fetch API -> CSV, download images, delete unused image files.
Run: python -m scripts.sync_products [--no-download] [--no-cleanup]
"""
import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

import pandas as pd
import requests

from skin_assistant.config import get_settings
from skin_assistant.infrastructure.skinme_client import (
    fetch_products,
    primary_image_url_from_row,
    products_to_csv,
    load_existing_csv,
    sync_products_to_csv,
)


def download_image(url: str, dest: Path, timeout: int = 15) -> bool:
    """Download one image to dest. Returns True on success."""
    try:
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception:
        return False


def download_product_images(csv_path: Path, images_dir: Path, timeout: int = 15) -> dict:
    """
    Download images for products in CSV. Filename: {id}_{image_id}.jpg (or from URL).
    Returns {downloaded: int, failed: int, skipped: int}.
    """
    df = pd.read_csv(csv_path)
    if df.empty or "id" not in df.columns:
        return {"downloaded": 0, "failed": 0, "skipped": 0}
    images_dir.mkdir(parents=True, exist_ok=True)
    downloaded = failed = skipped = 0
    for _, row in df.iterrows():
        url = primary_image_url_from_row(row.get("image_url"), row.get("all_image_urls"))
        if not url or pd.isna(url):
            skipped += 1
            continue
        pid = row.get("id")
        image_id = row.get("image_id", pid)
        ext = ".jpg"
        if ".png" in str(url).lower():
            ext = ".png"
        elif ".webp" in str(url).lower():
            ext = ".webp"
        elif ".avif" in str(url).lower():
            ext = ".avif"
        dest = images_dir / f"{pid}_{image_id}{ext}"
        if dest.exists():
            skipped += 1
            continue
        if download_image(url, dest, timeout=timeout):
            downloaded += 1
        else:
            failed += 1
    return {"downloaded": downloaded, "failed": failed, "skipped": skipped}


def cleanup_unused_images(csv_path: Path, images_dir: Path) -> dict:
    """
    Delete image files in images_dir that are not referenced by current product IDs in CSV.
    Returns {deleted: int, kept: int}.
    """
    if not images_dir.exists():
        return {"deleted": 0, "kept": 0}
    df = pd.read_csv(csv_path)
    valid_ids = set(df["id"].astype(str).unique()) if not df.empty and "id" in df.columns else set()
    deleted = kept = 0
    for f in list(images_dir.iterdir()):
        if not f.is_file():
            continue
        # filename like "1_1.jpg" or "6_7.avif" -> product id is first part
        try:
            pid = f.stem.split("_")[0]
        except Exception:
            pid = ""
        if pid in valid_ids:
            kept += 1
        else:
            try:
                f.unlink()
                deleted += 1
            except Exception:
                pass
    return {"deleted": deleted, "kept": kept}


def do_sync(
    no_download: bool = False,
    no_cleanup: bool = False,
    no_sync: bool = False,
    overwrite_existing: bool = False,
) -> int:
    """
    Run sync: fetch from backend -> CSV (if not existing or overwrite_existing), download only missing images, cleanup.
    When overwrite_existing=False, existing CSV is left as-is; images are always skipped if file already exists.
    """
    settings = get_settings()
    csv_path = settings.skinme_products_path
    images_dir = settings.product_images_dir

    if not no_sync:
        stats = sync_products_to_csv(csv_path=csv_path, overwrite_existing=overwrite_existing)
        print("Sync:", stats)
        if stats.get("error"):
            print("Warning:", stats.get("message", stats.get("error")))
        if csv_path.exists():
            total = int(stats.get("total", 0) or 0)
            print(f"CSV: {csv_path} ({total} products)" + (" (left existing)" if stats.get("skipped") else ""))
        else:
            print("No CSV written; sync failed or no data.")
            return 1

    if not no_download and csv_path.exists():
        dl = download_product_images(csv_path, images_dir)
        print("Images:", dl)

    if not no_cleanup and csv_path.exists() and images_dir.exists():
        cl = cleanup_unused_images(csv_path, images_dir)
        print("Cleanup:", cl)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Sync SkinMe products to CSV, download images, cleanup")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading product images")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip deleting unused image files")
    parser.add_argument("--no-sync", action="store_true", help="Skip API fetch (only download/cleanup with existing CSV)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV; default is to leave existing data")
    args = parser.parse_args()
    return do_sync(
        no_download=args.no_download,
        no_cleanup=args.no_cleanup,
        no_sync=args.no_sync,
        overwrite_existing=args.overwrite,
    )


if __name__ == "__main__":
    sys.exit(main())
