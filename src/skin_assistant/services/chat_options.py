"""Suggested quick-reply options for chat UIs (buttons / chips)."""
from __future__ import annotations

from typing import List


_DEFAULT_OPTIONS = [
    "Recommend products for dry skin",
    "What is niacinamide?",
    "Products for acne-prone skin",
    "I need help choosing skincare",
]

_GREETING_OPTIONS = [
    "Recommend products for dry skin",
    "What is hyaluronic acid?",
    "I have sensitive skin",
    "Connect me with support",
]

_INGREDIENT_OPTIONS = [
    "Products containing this ingredient",
    "Recommend for oily skin",
    "Recommend for dry skin",
    "Another skincare question",
]

_PRODUCT_OPTIONS = [
    "Show more product options",
    "Products under $30",
    "Ingredient details",
    "Connect me with support",
]


def get_suggested_options(user_message: str, reply: str = "") -> List[str]:
    """Return 3–4 short quick-reply labels based on the latest user turn."""
    msg = (user_message or "").strip().lower()
    combined = f"{msg} {(reply or '').lower()}"

    if any(w in msg for w in ("hello", "hi", "hey", "សួស្តី", "ជំរាប")):
        return _GREETING_OPTIONS[:4]

    if any(w in msg for w in ("ingredient", "what is", "what's", "niacinamide", "retinol", "hyaluronic")):
        return _INGREDIENT_OPTIONS[:4]

    if any(
        w in combined
        for w in (
            "recommend",
            "product",
            "dry skin",
            "oily",
            "acne",
            "sensitive",
            "budget",
            "$",
        )
    ):
        return _PRODUCT_OPTIONS[:4]

    if any(w in msg for w in ("admin", "human", "support", "agent", "staff", "person")):
        return [
            "Connect me with support",
            "Recommend products for my skin",
            "What is niacinamide?",
            "I have another question",
        ]

    return _DEFAULT_OPTIONS[:4]
