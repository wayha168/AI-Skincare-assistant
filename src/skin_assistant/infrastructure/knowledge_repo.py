"""Data layer: load and search ingredients and products from CSV files in data/ plus optional MySQL (skinme_db)."""
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from skin_assistant.config import get_settings
from skin_assistant.infrastructure.skinme_client import rewrite_skinme_product_image_urls


def _normalize_product_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _dedupe_products(products: list[dict]) -> list[dict]:
    """First occurrence wins (scraped CSV rows before DB when merged in that order)."""
    seen: set[str] = set()
    out: list[dict] = []
    for p in products:
        key = _normalize_product_name(str(p.get("product_name") or ""))
        if not key:
            out.append(p)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# Already loaded by dedicated methods — skip in auto folder scan
_SKIP_SUPPLEMENTARY_GLOB = frozenset({
    "ingredientsList.csv",
    "ingredient_lookup.csv",
    "products_ingredients_dataset.csv",
    "skincare_products_clean.csv",
    "skinme_products.csv",
    "Paula_SUM_LIST.csv",
    "Paula_embedding_SUMLIST_before_422.csv",
})


def _read_csv_flexible(path: Path) -> Optional[pd.DataFrame]:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", engine="python")
        except Exception:
            continue
    return None


def _norm_product_frame(
    df: pd.DataFrame,
    name_col: str,
    price_col: Optional[str],
    url_col: Optional[str],
    type_col: Optional[str],
    text_cols: list[str],
    source: str,
) -> pd.DataFrame:
    if df.empty or name_col not in df.columns:
        return pd.DataFrame()
    use = df.copy()
    use["_pn"] = use[name_col].fillna("").astype(str)
    use["_price"] = pd.Series([""] * len(use), index=use.index, dtype=object)
    if price_col and price_col in use.columns:
        use["_price"] = use[price_col].apply(lambda x: "" if pd.isna(x) else str(x))
    use["_url"] = pd.Series([""] * len(use), index=use.index, dtype=object)
    if url_col and url_col in use.columns:
        use["_url"] = use[url_col].fillna("").astype(str)
    use["_ptype"] = pd.Series([""] * len(use), index=use.index, dtype=object)
    if type_col and type_col in use.columns:
        use["_ptype"] = use[type_col].fillna("").astype(str)
    parts = [use["_pn"]]
    for c in text_cols:
        if c in use.columns and c != name_col:
            parts.append(use[c].fillna("").astype(str))
    search = pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower()
    n = len(use)
    return pd.DataFrame({
        "_pn": use["_pn"].astype(str).values,
        "_price": use["_price"].astype(str).values,
        "_url": use["_url"].astype(str).values,
        "_ptype": use["_ptype"].astype(str).values,
        "_search_text": search.values,
        "_source": [source] * n,
    })


def _load_supplementary_normalized(path: Path) -> pd.DataFrame:
    """Map one data/*.csv file to columns _pn, _price, _url, _ptype, _search_text, _source."""
    df = _read_csv_flexible(path)
    if df is None or df.empty:
        return pd.DataFrame()
    name = path.name

    if name == "Sephora_all_423.csv":
        text = [
            "ingredients", "about", "reviews", "recommended", "What it is", "Skin Type",
            "Skincare Concerns", "Formulation", "Benefits", "Highlighted Ingredients",
            "Ingredient Callouts", "What Else You Need to Know", "Clinical Results",
            "brand_name", "clean_ingredients", "new_ingredients",
        ]
        text = [c for c in text if c in df.columns]
        return _norm_product_frame(
            df, "cosmetic_name", "price", "cosmetic_link", "Formulation", text, name,
        )

    if name == "E-commerce  cosmetic dataset.csv":
        text = [c for c in df.columns if c not in ("product_name", "price", "website")]
        return _norm_product_frame(df, "product_name", "price", "website", "category", text, name)

    if name == "CELESTIA SKIN CARE DASTASET.csv":
        df = df.copy()
        c1 = df["Concern"].fillna("") if "Concern" in df.columns else pd.Series("", index=df.index)
        c2 = df["Internal_Type"].fillna("") if "Internal_Type" in df.columns else pd.Series("", index=df.index)
        df["_row_name"] = c1.astype(str) + " — " + c2.astype(str)
        text = [c for c in ("Skin_Type", "Skin_Subtype", "Sensitivity", "Concern", "Ingredients", "Effects", "Internal_Type") if c in df.columns]
        return _norm_product_frame(df, "_row_name", None, None, None, text, name)

    if name == "pre_alternatives.csv":
        df = df.copy()
        df["_pair"] = df["component1"].fillna("").astype(str) + " ↔ " + df["component2"].fillna("").astype(str)
        return _norm_product_frame(df, "_pair", None, None, None, ["component1", "component2"], name)

    if name == "binary_cosmetic_ingredient.csv":
        df = df.copy()
        df["_combo"] = df["cosmetic"].fillna("").astype(str) + " / " + df["ingredient"].fillna("").astype(str)
        return _norm_product_frame(df, "_combo", None, None, None, ["cosmetic", "ingredient"], name)

    if name == "db_import_template.csv":
        text = [
            c for c in (
                "product_name", "category", "concerns", "clean_ingredients", "key_ingredients",
                "ai_tags", "full_ingredients", "brand", "skin_types",
            ) if c in df.columns
        ]
        return _norm_product_frame(df, "product_name", "price_usd", None, "category", text, name)

    # Generic: first plausible name column + all object columns as search text
    name_guess = None
    for cand in ("product_name", "cosmetic_name", "name", "title", "cosmetic", "sku"):
        for c in df.columns:
            if cand.lower() in c.lower():
                name_guess = c
                break
        if name_guess:
            break
    if not name_guess:
        name_guess = df.columns[0]
    obj_cols = [c for c in df.columns if df[c].dtype == object or "string" in str(df[c].dtype)]
    price_guess = next((c for c in df.columns if "price" in c.lower()), None)
    url_guess = next((c for c in df.columns if "url" in c.lower() or "link" in c.lower() or "href" in c.lower()), None)
    return _norm_product_frame(df, name_guess, price_guess, url_guess, None, obj_cols, name)


class KnowledgeRepository:
    """Repository for ingredients (multiple CSVs) and products (scraped skinme_products.csv + optional MySQL)."""

    def __init__(self, data_dir: Optional[Path] = None):
        settings = get_settings()
        self._data_dir = data_dir or settings.data_dir
        self._ingredients_df: Optional[pd.DataFrame] = None
        self._ingredient_lookup_df: Optional[pd.DataFrame] = None
        self._products_df: Optional[pd.DataFrame] = None
        self._dataset_products_df: Optional[pd.DataFrame] = None
        self._skinme_df: Optional[pd.DataFrame] = None
        self._supplementary_products_df: Optional[pd.DataFrame] = None
        self._paula_ingredients_df: Optional[pd.DataFrame] = None
        self._mysql_client: Any = None
        self._mysql_tried = False

    def _mysql(self):
        """Lazy SkinMeDBClient when MYSQL_* env is set."""
        if not self._mysql_tried:
            self._mysql_tried = True
            try:
                from skin_assistant.infrastructure.skinme_db import SkinMeDBClient
                c = SkinMeDBClient()
                self._mysql_client = c if c.is_available() else None
            except Exception:
                self._mysql_client = None
        return self._mysql_client

    def _ensure_ingredients(self) -> pd.DataFrame:
        if self._ingredients_df is None:
            path = self._data_dir / "ingredientsList.csv"
            if not path.exists():
                self._ingredients_df = pd.DataFrame()
            else:
                df = pd.read_csv(path)
                cols = ["name", "short_description", "what_is_it", "what_does_it_do", "who_is_it_good_for", "who_should_avoid"]
                existing = [c for c in cols if c in df.columns]
                df["_search_text"] = df[existing].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
                self._ingredients_df = df
        return self._ingredients_df

    def _ensure_ingredient_lookup(self) -> pd.DataFrame:
        if self._ingredient_lookup_df is None:
            path = get_settings().ingredient_lookup_path
            if not path.exists():
                self._ingredient_lookup_df = pd.DataFrame()
            else:
                df = pd.read_csv(path)
                if "ingredient_name" in df.columns:
                    parts = [df["ingredient_name"].fillna("").astype(str)]
                    for c in ("primary_function", "typical_skin_types", "related_concerns", "clean_name"):
                        if c in df.columns:
                            parts.append(df[c].fillna("").astype(str))
                    df["_search_text"] = pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower()
                else:
                    df["_search_text"] = ""
                self._ingredient_lookup_df = df
        return self._ingredient_lookup_df

    def _ensure_products(self) -> pd.DataFrame:
        if self._products_df is None:
            path = self._data_dir / "skincare_products_clean.csv"
            if not path.exists():
                self._products_df = pd.DataFrame()
            else:
                df = pd.read_csv(path)
                if "clean_ingreds" in df.columns:
                    df["_ingreds_str"] = df["clean_ingreds"].fillna("").astype(str).str.lower()
                self._products_df = df
        return self._products_df

    def _ensure_dataset_products(self) -> pd.DataFrame:
        if self._dataset_products_df is None:
            path = get_settings().dataset_products_path
            if not path.exists():
                self._dataset_products_df = pd.DataFrame()
            else:
                df = pd.read_csv(path)
                text_cols = [
                    c
                    for c in (
                        "product_name",
                        "category",
                        "concerns",
                        "clean_ingredients",
                        "key_ingredients",
                        "ai_tags",
                        "full_ingredients",
                        "brand",
                    )
                    if c in df.columns
                ]
                if text_cols:
                    df["_search_text"] = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
                else:
                    df["_search_text"] = ""
                self._dataset_products_df = df
        return self._dataset_products_df

    def _ensure_skinme(self) -> pd.DataFrame:
        if self._skinme_df is None:
            path = get_settings().skinme_products_path
            if not path.exists():
                self._skinme_df = pd.DataFrame()
            else:
                df = pd.read_csv(path)
                df = rewrite_skinme_product_image_urls(df)
                parts = [
                    df.get("name", pd.Series("")).fillna(""),
                    df.get("description", pd.Series("")).fillna(""),
                    df.get("productType", pd.Series("")).fillna("").astype(str),
                    df.get("category_name", pd.Series("")).fillna("").astype(str),
                ]
                df["_search_text"] = pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower()
                self._skinme_df = df
        return self._skinme_df

    def _mask_concern_on_search_text(
        self, subset: pd.DataFrame, concern: str, text_col: str = "_search_text"
    ) -> pd.DataFrame:
        if subset.empty or not concern or not concern.strip():
            return subset.iloc[0:0]
        c = concern.strip().lower()
        if text_col not in subset.columns:
            return subset.iloc[0:0]
        mask = subset[text_col].str.contains(re.escape(c), na=False)
        if not mask.any():
            for t in set(re.findall(r"\w+", c)):
                if len(t) >= 3:
                    mask = mask | subset[text_col].str.contains(re.escape(t), na=False)
        return subset[mask]

    def _search_dataset_products_by_concern(
        self, concern: str, product_type: Optional[str] = None, top_k: int = 5
    ) -> list[dict]:
        df = self._ensure_dataset_products()
        if df.empty:
            return []
        subset = df
        if product_type and "category" in subset.columns:
            subset = subset[
                subset["category"].fillna("").astype(str).str.lower().str.contains(product_type.lower(), na=False)
            ]
        matched = self._mask_concern_on_search_text(subset, concern)
        out = []
        for _, row in matched.head(top_k).iterrows():
            price = row.get("price_usd", "")
            if price is None or (isinstance(price, float) and pd.isna(price)):
                price = ""
            out.append({
                "product_name": row.get("product_name"),
                "product_type": row.get("category") or "",
                "price": str(price) if price != "" else "",
                "product_url": "",
                "_source": "products_ingredients_dataset.csv",
            })
        return out

    def _search_skincare_csv_by_concern(
        self, concern: str, product_type: Optional[str] = None, top_k: int = 5
    ) -> list[dict]:
        df = self._ensure_products()
        if df.empty or not concern or not concern.strip():
            return []
        subset = df
        if product_type and "product_type" in subset.columns:
            subset = subset[
                subset["product_type"].fillna("").astype(str).str.lower().str.contains(product_type.lower(), na=False)
            ]
        if "_ingreds_str" not in subset.columns:
            return []
        c = concern.strip().lower()
        mask = subset["_ingreds_str"].str.contains(re.escape(c), na=False)
        name_match = subset["product_name"].fillna("").astype(str).str.lower().str.contains(re.escape(c), na=False)
        mask = mask | name_match
        if not mask.any():
            for t in set(re.findall(r"\w+", c)):
                if len(t) >= 3:
                    mask = mask | subset["_ingreds_str"].str.contains(re.escape(t), na=False)
        subset = subset[mask]
        out = []
        for _, row in subset.head(top_k).iterrows():
            out.append({
                "product_name": row.get("product_name"),
                "product_type": row.get("product_type") or "",
                "price": str(row.get("price") or ""),
                "product_url": row.get("product_url") or "",
                "_source": "skincare_products_clean.csv",
            })
        return out

    def _search_skinme_by_concern(
        self, concern: str, product_type: Optional[str] = None, top_k: int = 5
    ) -> list[dict]:
        df = self._ensure_skinme()
        if df.empty or not concern or not concern.strip():
            return []
        subset = df
        if product_type:
            subset = subset[
                subset["productType"].fillna("").astype(str).str.lower().str.contains(product_type.lower(), na=False)
            ]
        matched = self._mask_concern_on_search_text(subset, concern)
        out = []
        for _, row in matched.head(top_k).iterrows():
            product_id = row.get("id")
            product_details_url = None
            if product_id:
                product_details_url = f"https://skinme.store/product_details?productId={product_id}"
            out.append({
                "product_name": row.get("name"),
                "product_type": row.get("productType"),
                "price": str(row.get("price", "")),
                "product_url": row.get("image_url") or "",
                "product_id": product_id,
                "product_details_url": product_details_url,
                "_source": "skinme_products.csv",
            })
        return out

    def _ensure_supplementary_products(self) -> pd.DataFrame:
        """All other *.csv in data/ with product-like rows (Sephora, E-commerce, CELESTIA, etc.)."""
        if self._supplementary_products_df is None:
            frames: list[pd.DataFrame] = []
            for path in sorted(self._data_dir.glob("*.csv")):
                if path.name in _SKIP_SUPPLEMENTARY_GLOB:
                    continue
                try:
                    block = _load_supplementary_normalized(path)
                    if not block.empty:
                        frames.append(block)
                except Exception:
                    continue
            self._supplementary_products_df = (
                pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            )
        return self._supplementary_products_df

    def _search_supplementary_products_by_concern(
        self, concern: str, product_type: Optional[str] = None, top_k: int = 5
    ) -> list[dict]:
        df = self._ensure_supplementary_products()
        if df.empty or "_search_text" not in df.columns:
            return []
        subset = df
        if product_type and "_ptype" in subset.columns:
            subset = subset[
                subset["_ptype"].fillna("").astype(str).str.lower().str.contains(product_type.lower(), na=False)
            ]
        matched = self._mask_concern_on_search_text(subset, concern)
        out = []
        for _, row in matched.head(top_k).iterrows():
            out.append({
                "product_name": row.get("_pn") or "",
                "product_type": str(row.get("_ptype") or ""),
                "price": str(row.get("_price") or ""),
                "product_url": str(row.get("_url") or ""),
                "_source": str(row.get("_source") or "data/*.csv"),
            })
        return out

    def _ensure_paula_ingredients(self) -> pd.DataFrame:
        """Paula's Choice–style ingredient lists (text columns only; no embedding blobs)."""
        if self._paula_ingredients_df is None:
            frames: list[pd.DataFrame] = []
            for fn in ("Paula_SUM_LIST.csv", "Paula_embedding_SUMLIST_before_422.csv"):
                p = self._data_dir / fn
                if not p.exists():
                    continue
                try:
                    if "embedding" in fn.lower():
                        df = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip", engine="python")
                        if "all" in df.columns:
                            df = df.drop(columns=["all"], errors="ignore")
                    else:
                        df = _read_csv_flexible(p)
                    if df is None or df.empty or "ingredient_name" not in df.columns:
                        continue
                    parts = [df["ingredient_name"].fillna("").astype(str)]
                    for c in ("functions", "benefits", "categories", "description", "combined_text", "glance", "rating", "references"):
                        if c in df.columns:
                            parts.append(df[c].fillna("").astype(str))
                    df = df.copy()
                    df["_search_text"] = pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower()
                    df["_paula_src"] = fn
                    frames.append(df)
                except Exception:
                    continue
            self._paula_ingredients_df = (
                pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            )
        return self._paula_ingredients_df

    def search_ingredients(self, query: str, top_k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []
        q = query.strip().lower()
        out: list[dict] = []
        seen_names: set[str] = set()

        df = self._ensure_ingredients()
        if not df.empty:
            name_match = df["name"].fillna("").astype(str).str.lower().str.contains(re.escape(q), na=False)
            text_match = df["_search_text"].str.contains(re.escape(q), na=False)
            mask = name_match | text_match
            if not mask.any():
                for t in set(re.findall(r"\w+", q)):
                    if len(t) >= 3:
                        mask = mask | df["_search_text"].str.contains(re.escape(t), na=False)
            for _, row in df[mask].head(top_k).iterrows():
                rec = row.to_dict()
                nm = str(rec.get("name") or "").strip().lower()
                if nm:
                    seen_names.add(nm)
                out.append(rec)

        # Supplement from ingredient_lookup.csv (same data folder)
        lk = self._ensure_ingredient_lookup()
        if not lk.empty and "ingredient_name" in lk.columns and "_search_text" in lk.columns:
            mask = lk["_search_text"].str.contains(re.escape(q), na=False)
            if not mask.any():
                for t in set(re.findall(r"\w+", q)):
                    if len(t) >= 3:
                        mask = mask | lk["_search_text"].str.contains(re.escape(t), na=False)
            for _, row in lk[mask].iterrows():
                nm = str(row.get("ingredient_name") or "").strip().lower()
                if nm and nm in seen_names:
                    continue
                if nm:
                    seen_names.add(nm)
                out.append({
                    "name": row.get("ingredient_name"),
                    "what_is_it": row.get("primary_function") or row.get("clean_name") or "",
                    "what_does_it_do": str(row.get("related_concerns") or ""),
                    "who_is_it_good_for": str(row.get("typical_skin_types") or ""),
                    "who_should_avoid": str(row.get("pregnancy_note") or ""),
                    "short_description": "",
                })
                if len(out) >= top_k:
                    break

        # Paula_SUM_LIST + Paula_embedding (text columns) in data/
        paula = self._ensure_paula_ingredients()
        if not paula.empty and "ingredient_name" in paula.columns and "_search_text" in paula.columns:
            mask = paula["_search_text"].str.contains(re.escape(q), na=False)
            if not mask.any():
                for t in set(re.findall(r"\w+", q)):
                    if len(t) >= 3:
                        mask = mask | paula["_search_text"].str.contains(re.escape(t), na=False)
            for _, row in paula[mask].iterrows():
                nm = str(row.get("ingredient_name") or "").strip().lower()
                if nm and nm in seen_names:
                    continue
                if nm:
                    seen_names.add(nm)
                out.append({
                    "name": row.get("ingredient_name"),
                    "what_is_it": str(row.get("functions") or row.get("rating") or ""),
                    "what_does_it_do": str(row.get("benefits") or row.get("description") or row.get("combined_text") or "")[:2000],
                    "who_is_it_good_for": str(row.get("categories") or ""),
                    "who_should_avoid": "",
                    "short_description": str(row.get("glance") or ""),
                })
                if len(out) >= top_k:
                    break

        return out[:top_k]

    def get_ingredient_by_name(self, name: str) -> Optional[dict]:
        if not name or not name.strip():
            return None
        df = self._ensure_ingredients()
        if not df.empty:
            n = name.strip().lower()
            for _, row in df.iterrows():
                if row.get("name") and str(row["name"]).lower() == n:
                    return row.to_dict()
            name_match = df["name"].fillna("").astype(str).str.lower().str.contains(re.escape(n), na=False)
            if name_match.any():
                return df[name_match].iloc[0].to_dict()
        lk = self._ensure_ingredient_lookup()
        if not lk.empty and "ingredient_name" in lk.columns:
            n = name.strip().lower()
            for _, row in lk.iterrows():
                if str(row.get("ingredient_name") or "").lower() == n:
                    return {
                        "name": row.get("ingredient_name"),
                        "what_is_it": row.get("primary_function") or "",
                        "what_does_it_do": str(row.get("related_concerns") or ""),
                        "who_is_it_good_for": str(row.get("typical_skin_types") or ""),
                        "who_should_avoid": str(row.get("pregnancy_note") or ""),
                    }
            nm = lk["ingredient_name"].fillna("").astype(str).str.lower().str.contains(re.escape(n), na=False)
            if nm.any():
                row = lk[nm].iloc[0]
                return {
                    "name": row.get("ingredient_name"),
                    "what_is_it": row.get("primary_function") or "",
                    "what_does_it_do": str(row.get("related_concerns") or ""),
                    "who_is_it_good_for": str(row.get("typical_skin_types") or ""),
                    "who_should_avoid": str(row.get("pregnancy_note") or ""),
                }
        paula = self._ensure_paula_ingredients()
        if not paula.empty and "ingredient_name" in paula.columns:
            n = name.strip().lower()
            for _, row in paula.iterrows():
                if str(row.get("ingredient_name") or "").lower() == n:
                    return {
                        "name": row.get("ingredient_name"),
                        "what_is_it": str(row.get("functions") or ""),
                        "what_does_it_do": str(row.get("benefits") or row.get("description") or row.get("combined_text") or ""),
                        "who_is_it_good_for": str(row.get("categories") or ""),
                        "who_should_avoid": "",
                    }
            sub = paula["ingredient_name"].fillna("").astype(str).str.lower().str.contains(re.escape(n), na=False)
            if sub.any():
                row = paula[sub].iloc[0]
                return {
                    "name": row.get("ingredient_name"),
                    "what_is_it": str(row.get("functions") or ""),
                    "what_does_it_do": str(row.get("benefits") or row.get("description") or row.get("combined_text") or ""),
                    "who_is_it_good_for": str(row.get("categories") or ""),
                    "who_should_avoid": "",
                }
        return None

    def list_indexed_csv_sources(self) -> dict:
        """Which CSV files under data/ feed search (for debugging / docs)."""
        core = [
            "ingredientsList.csv",
            "ingredient_lookup.csv",
            "products_ingredients_dataset.csv",
            "skincare_products_clean.csv",
            "skinme_products.csv",
        ]
        supp = sorted(
            p.name for p in self._data_dir.glob("*.csv") if p.name not in _SKIP_SUPPLEMENTARY_GLOB
        )
        paula = [
            "Paula_SUM_LIST.csv",
            "Paula_embedding_SUMLIST_before_422.csv",
        ]
        return {
            "core_product_and_ingredient": core,
            "additional_product_rows_from": supp,
            "additional_ingredient_rows_from": [f for f in paula if (self._data_dir / f).exists()],
            "product_recommendations_use": "skinme_products.csv (scraped) + MySQL product table when use_database=true",
        }

    def search_products_by_concern(
        self, concern: str, product_type: Optional[str] = None, top_k: int = 5, use_database: bool = False
    ) -> list[dict]:
        """
        Merged products: scraped skinme_products.csv first, then MySQL (skinme_db) when use_database=True and MYSQL_* is set.
        """
        if not concern or not concern.strip():
            return []

        pool: list[dict] = []
        csv_rows = self._search_skinme_by_concern(concern, product_type=product_type, top_k=top_k)
        for p in csv_rows:
            p.pop("_source", None)
            pool.append(p)
        if use_database:
            db = self._mysql()
            if db:
                pool.extend(db.search_products_by_concern(concern, product_type=product_type, top_k=top_k))
        return _dedupe_products(pool)[:top_k]

    def search_skinme_products_for_chat_context(
        self,
        concern_query: str,
        ingredient_hits: list[dict],
        product_type: Optional[str] = None,
        top_k: int = 8,
        use_database: bool = False,
    ) -> list[dict]:
        """
        Aligned to ingredient context: scraped CSV + optional DB. Ingredient facts still come from all ingredient CSVs.
        """
        concern_query = (concern_query or "").strip()
        if not concern_query and not ingredient_hits:
            return []

        if not ingredient_hits:
            return self.search_products_by_concern(
                concern_query, product_type=product_type, top_k=top_k, use_database=use_database
            )

        seen: set[str] = set()
        out: list[dict] = []

        def _add(p: dict) -> None:
            name = p.get("product_name")
            if name and name not in seen:
                seen.add(name)
                out.append(p)

        for h in ingredient_hits[:12]:
            iname = (h.get("name") or "").strip()
            if len(iname) < 2:
                continue
            for p in self.get_products_containing_ingredient(
                iname, top_k=top_k, use_database=use_database
            ):
                _add(p)
                if len(out) >= top_k:
                    return out[:top_k]

        if concern_query and len(out) < top_k:
            need = top_k - len(out)
            for row in self._search_skinme_by_concern(
                concern_query, product_type=product_type, top_k=need + len(seen) + 8
            ):
                row.pop("_source", None)
                _add(row)
                if len(out) >= top_k:
                    break
            if use_database and len(out) < top_k:
                db = self._mysql()
                if db:
                    for p in db.search_products_by_concern(
                        concern_query, product_type=product_type, top_k=need + 8
                    ):
                        _add(p)
                        if len(out) >= top_k:
                            break

        return out[:top_k]

    def get_products_containing_ingredient(
        self, ingredient_name: str, top_k: int = 5, use_database: bool = False
    ) -> list[dict]:
        """Scraped skinme_products.csv rows, plus MySQL matches when use_database=True."""
        if not ingredient_name or not ingredient_name.strip():
            return []
        key = ingredient_name.strip().lower()
        pool: list[dict] = []
        df = self._ensure_skinme()
        if not df.empty and "_search_text" in df.columns:
            mask = df["_search_text"].str.contains(re.escape(key), na=False)
            if not mask.any():
                for t in set(re.findall(r"\w+", key)):
                    if len(t) >= 3:
                        mask = mask | df["_search_text"].str.contains(re.escape(t), na=False)
            for _, row in df[mask].head(top_k).iterrows():
                product_id = row.get("id")
                product_details_url = None
                if product_id:
                    product_details_url = f"https://skinme.store/product_details?productId={product_id}"
                pool.append({
                    "product_name": row.get("name"),
                    "product_type": row.get("productType"),
                    "price": str(row.get("price", "")),
                    "product_url": row.get("image_url") or "",
                    "product_id": product_id,
                    "product_details_url": product_details_url,
                })
        if use_database:
            db = self._mysql()
            if db:
                pool.extend(
                    db.search_products_by_ingredient(ingredient_name.strip(), top_k=top_k)
                )
        return _dedupe_products(pool)[:top_k]
