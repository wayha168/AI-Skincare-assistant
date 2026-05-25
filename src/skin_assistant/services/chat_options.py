"""Suggested quick-reply options for chat UIs (buttons / chips)."""
from __future__ import annotations

from typing import List


_DEFAULT_OPTIONS = [
    "Ask about skin concerns",
    "What is niacinamide?",
    "Ask about acne solutions",
    "I need help choosing skincare",
]

_GREETING_OPTIONS = [
    "How can I help you with your skin?",
    "What ingredients are good for skin tone?",
    "I have sensitive skin",
    "Connect me with support",
]

_INGREDIENT_OPTIONS = [
    "What does this ingredient do?",
    "Which skin types is it good for?",
    "Who should avoid this ingredient?",
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
            "suggest",
            "product",
            "show",
            "options",
        )
    ):
        return _PRODUCT_OPTIONS[:4]

    if any(w in msg for w in ("admin", "human", "support", "agent", "staff", "person")):
        return [
            "Connect me with support",
            "Ask about skin concerns",
            "What is niacinamide?",
            "I have another question",
        ]

    return _DEFAULT_OPTIONS[:4]