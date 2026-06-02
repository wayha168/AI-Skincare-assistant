"""
Optional MySQL (skinme_db) client for product search when "use database" is enabled in chat.
Credentials from environment: MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE.
"""
from typing import List, Optional

from skin_assistant.config import get_settings


def _row_to_product(row: dict) -> dict:
    """Map DB row (camelCase or snake_case) to API shape: product_name, product_type, price, product_url, product_id, product_details_url."""
    product_id = row.get("id") or row.get("product_id")
    product_details_url = None
    if product_id:
        product_details_url = f"https://skinme.store/product_details?productId={product_id}"
    
    return {
        "product_name": row.get("product_name") or row.get("name"),
        "product_type": row.get("product_type") or row.get("productType"),
        "price": str(row.get("price") or ""),
        "product_url": row.get("product_url") or row.get("image_url") or row.get("imageUrl") or "",
        "product_id": product_id,
        "product_details_url": product_details_url,
    }


class SkinMeDBClient:
    """Query products from skinme_db (MySQL) when env is configured."""

    def __init__(self):
        self._settings = get_settings()
        self._conn = None

    def is_available(self) -> bool:
        return bool(self._settings.use_mysql_db)

    def _get_connection(self):
        if not self.is_available():
            return None
        try:
            import pymysql
        except ImportError:
            return None
        if self._conn is not None:
            try:
                self._conn.ping(reconnect=True)
                return self._conn
            except Exception:
                self._conn = None
        try:
            self._conn = pymysql.connect(
                host=self._settings.mysql_host,
                port=self._settings.mysql_port,
                user=self._settings.mysql_user,
                password=self._settings.mysql_password,
                database=self._settings.mysql_database,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                connect_timeout=5,
            )
            return self._conn
        except Exception:
            return None

    def search_products_by_concern(
        self, concern: str, product_type: Optional[str] = None, top_k: int = 5
    ) -> List[dict]:
        """
        Search products by concern (name, description, productType).
        Returns list of dicts with product_name, product_type, price, product_url.
        """
        conn = self._get_connection()
        if not conn:
            return []
        table = self._settings.mysql_products_table
        # Escape table name (no user input in table name when from env)
        safe_table = "`" + table.replace("`", "``") + "`"
        # Support both snake_case and camelCase column names
        like_arg = f"%{concern}%"
        try:
            with conn.cursor() as cur:
                # Match concern in name/description (common in Spring/JPA product table)
                sql = f"SELECT * FROM {safe_table} WHERE (name LIKE %s OR description LIKE %s) LIMIT %s"
                cur.execute(sql, (like_arg, like_arg, top_k))
                rows = cur.fetchall()
            return [_row_to_product(dict(r)) for r in rows]
        except Exception:
            return []
        finally:
            # Do not close conn; reuse for next call
            pass

    def search_products_by_ingredient(self, ingredient: str, top_k: int = 5) -> List[dict]:
        """Products whose name or description mentions the ingredient (for chat alignment with CSV scrape)."""
        conn = self._get_connection()
        if not conn or not ingredient or not ingredient.strip():
            return []
        table = self._settings.mysql_products_table
        safe_table = "`" + table.replace("`", "``") + "`"
        like_arg = f"%{ingredient.strip()}%"
        try:
            with conn.cursor() as cur:
                sql = f"SELECT * FROM {safe_table} WHERE (name LIKE %s OR description LIKE %s) LIMIT %s"
                cur.execute(sql, (like_arg, like_arg, top_k))
                rows = cur.fetchall()
            return [_row_to_product(dict(r)) for r in rows]
        except Exception:
            return []

    def close(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
