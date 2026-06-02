"""
Fetch product data from SkinMe API (JSON). Saves to CSV.
Image links use the public storefront host (e.g. https://skinme.store/uploads/...) so they match the frontend.
"""
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from skin_assistant.config import get_settings


def public_image_url(url: Optional[str]) -> str:
    """
    Normalize product image URLs to the public storefront (e.g. skinme.store/uploads/...).
    Rewrites absolute backend.skinme.store URLs to the configured frontend base.
    """
    if url is None:
        return ""
    s = str(url).strip()
    if not s:
        return ""
    settings = get_settings()
    front = settings.skinme_frontend_base_url.rstrip("/")
    backend = settings.skinme_base_url.rstrip("/")
    if s.startswith(backend + "/") or s == backend:
        return front + s[len(backend) :]
    if s.startswith("/"):
        return front + s
    return s


def _full_image_url(img: dict) -> str:
    download_url = (img or {}).get("downloadUrl") or ""
    if not download_url:
        return ""
    settings = get_settings()
    front = settings.skinme_frontend_base_url.rstrip("/")
    backend = settings.skinme_base_url.rstrip("/")
    if download_url.startswith("/"):
        return front + download_url
    if download_url.startswith(backend + "/") or download_url == backend:
        return front + download_url[len(backend) :]
    return public_image_url(download_url)


def _deduped_image_urls(images: list) -> list[str]:
    """Ordered unique image URLs from API images array (no duplicate URLs)."""
    seen: set[str] = set()
    out: list[str] = []
    for img in images or []:
        full = _full_image_url(img if isinstance(img, dict) else {})
        if not full or full in seen:
            continue
        seen.add(full)
        out.append(full)
    return out


def primary_image_url_from_row(image_url: object, all_image_urls: object) -> str:
    """
    Primary image for a CSV/DB row: prefer image_url, else first segment of all_image_urls.
    Supports legacy CSV where all_image_urls listed every image including the first.
    """
    if image_url is not None and not pd.isna(image_url) and str(image_url).strip():
        return str(image_url).strip()
    if all_image_urls is None or pd.isna(all_image_urls):
        return ""
    for part in str(all_image_urls).split("|"):
        p = part.strip()
        if p:
            return p
    return ""


def rewrite_skinme_product_image_urls(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite image_url / all_image_urls in a SkinMe products DataFrame to storefront URLs."""
    if df.empty:
        return df
    out = df.copy()
    if "image_url" in out.columns:
        out["image_url"] = out["image_url"].apply(
            lambda x: public_image_url(str(x) if pd.notna(x) else "") or ""
        )
    if "all_image_urls" in out.columns:
        def _pipe(s: object) -> object:
            if pd.isna(s) or not str(s).strip():
                return s
            parts = [public_image_url(p.strip()) for p in str(s).split("|") if p.strip()]
            return "|".join(parts)
        out["all_image_urls"] = out["all_image_urls"].apply(_pipe)
    return out


def all_image_urls_for_row(image_url: object, all_image_urls: object) -> list[str]:
    """
    Full ordered list: primary (image_url) plus additional gallery URLs (all_image_urls).
    Dedupes when legacy rows repeat the primary inside all_image_urls.
    """
    primary = primary_image_url_from_row(image_url, "")
    extras = "" if all_image_urls is None or pd.isna(all_image_urls) else str(all_image_urls)
    out: list[str] = []
    if primary:
        out.append(primary)
    for part in extras.split("|"):
        p = part.strip()
        if not p:
            continue
        if p not in out:
            out.append(p)
    return out


def _normalize_product(p: dict) -> dict:
    """Flatten one API product into a CSV row."""
    cat = p.get("category") or {}
    images = p.get("images") or []
    first_img = images[0] if images else {}
    urls = _deduped_image_urls(images)
    primary = urls[0] if urls else ""
    # all_image_urls = gallery only (no repeat of image_url) so the DB/CSV stays clean
    additional = "|".join(urls[1:])
    return {
        "id": p.get("id"),
        "name": p.get("name"),
        "brand": p.get("brand"),
        "price": p.get("price"),
        "productType": p.get("productType"),
        "inventory": p.get("inventory"),
        "description": p.get("description"),
        "howToUse": p.get("howToUse"),
        "category_id": cat.get("id"),
        "category_name": cat.get("name"),
        "image_url": primary,
        "image_id": first_img.get("imageId"),
        "image_filename": first_img.get("fileName"),
        "all_image_urls": additional,
    }


def fetch_products(api_url: Optional[str] = None, timeout: int = 30) -> list[dict]:
    """
    Fetch all products from SkinMe API. Returns list of normalized product dicts.
    Uses requests (JSON API); use bs4 only when scraping HTML pages.
    Raises requests.RequestException on network failure, ValueError on bad JSON shape.
    """
    url = api_url or get_settings().skinme_api_url
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get("message") != "success" or "data" not in data:
        raise ValueError("Unexpected API response: missing data")
    return [_normalize_product(p) for p in data["data"]]


def products_to_csv(products: list[dict], csv_path: Path) -> None:
    """Write products to CSV."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(products)
    df.to_csv(csv_path, index=False, encoding="utf-8")


def load_existing_csv(csv_path: Path) -> pd.DataFrame:
    """Load existing CSV if present; empty DataFrame otherwise."""
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def sync_products_to_csv(
    csv_path: Optional[Path] = None,
    api_url: Optional[str] = None,
    overwrite_existing: bool = False,
) -> dict:
    """
    Fetch products from API and sync to CSV with intelligent merging.
    
    When overwrite_existing=False and CSV exists:
    - Fetches from backend
    - Merges: keeps existing products (by ID), adds new ones automatically
    - Avoids re-scraping existing products while ensuring new products are added
    
    When overwrite_existing=True:
    - Fetches and overwrites entire CSV (full refresh)
    
    Returns stats: total, added, removed, path, skipped, optional error/message.
    """
    settings = get_settings()
    csv_path = csv_path or settings.skinme_products_path
    old_df = load_existing_csv(csv_path)

    try:
        products = fetch_products(api_url=api_url)
    except requests.RequestException as e:
        n_existing = len(old_df) if not old_df.empty else 0
        return {
            "total": n_existing,
            "added": 0,
            "removed": 0,
            "path": str(csv_path),
            "skipped": bool(n_existing),
            "error": type(e).__name__,
            "message": f"Product API unreachable ({str(e)}). Using existing CSV if present.",
        }
    except ValueError as e:
        n_existing = len(old_df) if not old_df.empty else 0
        return {
            "total": n_existing,
            "added": 0,
            "removed": 0,
            "path": str(csv_path),
            "skipped": bool(n_existing),
            "error": "ValueError",
            "message": str(e),
        }

    old_ids = set(old_df["id"].astype(str).unique()) if not old_df.empty and "id" in old_df.columns else set()
    new_ids = {str(p["id"]) for p in products}
    added = len(new_ids - old_ids)
    removed = len(old_ids - new_ids)

    # Smart merge: if overwrite_existing=False and we have existing data, merge instead of overwrite
    if not overwrite_existing and not old_df.empty and "id" in old_df.columns:
        # Keep existing products, add new ones
        existing_ids_set = set(old_df["id"].astype(str).unique())
        new_products = [p for p in products if str(p["id"]) not in existing_ids_set]
        
        if new_products:
            # Convert to DataFrame and merge
            new_df = pd.DataFrame(new_products)
            merged_df = pd.concat([old_df, new_df], ignore_index=True)
            merged_df = rewrite_skinme_product_image_urls(merged_df)
            merged_df.to_csv(csv_path, index=False)
            return {
                "total": len(merged_df),
                "added": len(new_products),
                "removed": 0,
                "path": str(csv_path),
                "skipped": False,
                "message": f"Merged: kept {len(old_df)} existing products, added {len(new_products)} new ones.",
            }
        else:
            # No new products, return early
            return {
                "total": len(old_df),
                "added": 0,
                "removed": 0,
                "path": str(csv_path),
                "skipped": True,
                "message": "No new products found. Keeping existing data.",
            }
    else:
        # Full overwrite mode
        products_to_csv(products, csv_path)
        return {
            "total": len(products),
            "added": added,
            "removed": removed,
            "path": str(csv_path),
            "skipped": False,
        }
