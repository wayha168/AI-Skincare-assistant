"""Shared admin authentication for REST and WebSocket admin endpoints."""
import os
from typing import Optional


def admin_key_required() -> Optional[str]:
    """Return configured admin key, or None if admin endpoints are open (dev only)."""
    key = (os.environ.get("WS_ADMIN_KEY") or os.environ.get("ADMIN_API_KEY") or "").strip()
    return key or None


def is_admin_authenticated(admin_key: Optional[str]) -> bool:
    required = admin_key_required()
    if not required:
        return True
    return bool(admin_key) and admin_key == required
