"""Chat service: retrieval + optional LLM for skin assistant replies."""
import ast
import os
import re
from typing import Any, Optional

from skin_assistant.infrastructure import KnowledgeRepository


def _normalize_assistant_plain_text(text: str) -> str:
    """Remove markdown bold so replies read like natural voice (plain text, Gemini-style)."""
    if not text:
        return text
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    return text.replace("**", "").strip()


# Khmer script (Unicode Khmer + Khmer Symbols) — for language detection
_KHMER_SCRIPT_RE = re.compile(r"[\u1780-\u17FF\u19E0-\u19FF]")


def _message_uses_khmer(text: str) -> bool:
    """True if the message contains Khmer script (ភាសាខ្មែរ)."""
    if not text or not text.strip():
        return False
    return bool(_KHMER_SCRIPT_RE.search(text))


def _asks_reply_in_khmer(text: str) -> bool:
    """True when the user asks for a Khmer reply in English (no Khmer script yet)."""
    if not text:
        return False
    low = text.lower()
    if "ភាសាខ្មែរ" in text:
        return True
    return any(
        p in low
        for p in (
            "in khmer",
            "speak khmer",
            "reply in khmer",
            "answer in khmer",
            "respond in khmer",
        )
    )


def _prefer_khmer_reply(user_message: str, conversation_history: Optional[list[dict]] = None) -> bool:
    """
    Prefer Khmer for the assistant reply when the user uses Khmer, asks for Khmer,
    or recently wrote in Khmer (so short follow-ups like \"ok\" can stay in thread language).
    """
    if _message_uses_khmer(user_message):
        return True
    if _asks_reply_in_khmer(user_message):
        return True
    for m in (conversation_history or [])[-8:]:
        if m.get("role") == "user" and _message_uses_khmer((m.get("content") or "")):
            return True
    return False


def _gemini_api_key() -> Optional[str]:
    """Google AI API key for Gemini (GEMINI_API_KEY or GOOGLE_API_KEY)."""
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip() or None


def _llm_enabled() -> bool:
    """Gemini is used when a key is set and GOOGLE_API_ENABLED is not false (Spring-style toggle)."""
    if os.environ.get("GOOGLE_API_ENABLED", "").strip().lower() in ("0", "false", "no"):
        return False
    return bool(_gemini_api_key())

CONCERN_TYPES = {
    "moistur": "moisturiser",
    "dry": "moisturiser",
    "hydrat": "moisturiser",
    "acne": "moisturiser",
    "oil": "moisturiser",
    "sensitive": "moisturiser",
    "redness": "moisturiser",
    "aging": "moisturiser",
    "wrinkle": "moisturiser",
    "pigment": "moisturiser",
    "spf": "moisturiser",
    "sun": "moisturiser",
}


def _detect_concern_type(text: str) -> Optional[str]:
    t = text.lower()
    for key, ptype in CONCERN_TYPES.items():
        if key in t:
            return ptype
    return None


def _is_recommendation_request(text: str) -> bool:
    """True when user is asking for product recommendations — reply must use DB/synced products only."""
    msg = text.strip().lower()
    return any(
        w in msg
        for w in (
            "recommend",
            "recommendation",
            "suggest",
            "suggestion",
            "products for",
            "product for",
            "for my skin",
            "for dry",
            "for oily",
            "for acne",
            "for sensitive",
        )
    )


# Follow-up questions when user mentions a skin problem vaguely (like a real person would ask)
CONCERN_FOLLOWUPS = {
    "acne": (
        "I'd like to recommend the right things for you. Can you tell me a bit more?\n\n"
        "• What type of acne do you have — blackheads, whiteheads, cystic, or hormonal breakouts?\n"
        "• How severe is it — mild, moderate, or more persistent?\n"
        "• What's your skin type — oily, combination, or sensitive?\n\n"
        "Once I know this, I can suggest ingredients and products that fit your situation."
    ),
    "dry": (
        "To help you better, it would help to know:\n\n"
        "• Is your skin mostly dry (tight or flaky), or also sensitive or reactive?\n"
        "• Which areas are most affected — face, body, or both?\n"
        "• Do you have any other concerns (e.g. redness, dullness)?\n\n"
        "Then I can suggest ingredients and products that suit you."
    ),
    "oil": (
        "Quick questions so I can tailor my suggestions:\n\n"
        "• Is the oiliness all over or mainly in the T-zone (forehead, nose)?\n"
        "• Do you also have breakouts or enlarged pores?\n"
        "• Is your skin sensitive or quite tolerant to products?\n\n"
        "With that, I can recommend the right products for you."
    ),
    "sensitive": (
        "I'd like to suggest things that won't irritate you. Can you share:\n\n"
        "• What tends to trigger your skin — certain products, weather, or stress?\n"
        "• Do you get redness, itching, or both?\n"
        "• Any other concerns (dryness, breakouts) alongside sensitivity?\n\n"
        "Then I can point you to gentle ingredients and products."
    ),
    "redness": (
        "To give you useful advice:\n\n"
        "• Is the redness ongoing or does it flare up sometimes?\n"
        "• Do you have sensitive or reactive skin, or rosacea?\n"
        "• Which areas are affected — cheeks, nose, or all over?\n\n"
        "I can then suggest calming ingredients and products."
    ),
    "aging": (
        "To recommend the right products:\n\n"
        "• What bothers you most — fine lines, deeper wrinkles, dullness, or loss of firmness?\n"
        "• Do you have sensitive skin or use retinoids already?\n"
        "• Any other concerns (dryness, pigmentation)?\n\n"
        "Then I can suggest ingredients and products that fit your goals."
    ),
    "wrinkle": (
        "To recommend the right products:\n\n"
        "• What bothers you most — fine lines, deeper wrinkles, dullness, or loss of firmness?\n"
        "• Do you have sensitive skin or use retinoids already?\n"
        "• Any other concerns (dryness, pigmentation)?\n\n"
        "Then I can suggest ingredients and products that fit your goals."
    ),
}


def _get_concern_followup(user_message: str, conversation_history: Optional[list[dict]] = None) -> Optional[str]:
    """
    If the user mentions a skin problem vaguely (e.g. 'I have acne' without asking for products),
    return a friendly follow-up question to understand their problem type. Otherwise return None.
    """
    history = conversation_history or []
    # Continue the thread: do not repeat vague follow-ups after the assistant has already replied
    if history and any(m.get("role") == "assistant" for m in history):
        return None
    msg = user_message.strip().lower()
    # Photo analysis already encodes the condition — use retrieval/LLM with DB instead of vague follow-up
    if "from my skin photo the condition appears to be " in msg:
        return None
    if any(
        w in msg
        for w in (
            "recommend",
            "suggest",
            "product",
            "products",
            "what can i use",
            "what should i use",
            "give me",
            "find me",
            "best for",
            "good for",
            "help with",
            "to treat",
            "to fix",
        )
    ):
        return None
    concern_words = ["acne", "dry", "oil", "oily", "sensitive", "redness", "aging", "wrinkle", "pigment", "hydrat"]
    has_concern = any(c in msg for c in concern_words)
    is_short = len(msg) < 100
    is_statement = any(
        msg.startswith(p)
        for p in (
            "i have",
            "i got",
            "my skin",
            "i suffer",
            "suffering",
            "dealing with",
            "struggling",
            "help",
            "problem",
            "issues with",
            "concern",
        )
    ) or not msg.endswith("?")
    if has_concern and is_short and is_statement:
        if "acne" in msg:
            return CONCERN_FOLLOWUPS["acne"]
        if "dry" in msg or "hydrat" in msg:
            return CONCERN_FOLLOWUPS["dry"]
        if "oil" in msg:
            return CONCERN_FOLLOWUPS["oil"]
        if "sensitive" in msg:
            return CONCERN_FOLLOWUPS["sensitive"]
        if "redness" in msg:
            return CONCERN_FOLLOWUPS["redness"]
        if "aging" in msg or "wrinkle" in msg:
            return CONCERN_FOLLOWUPS.get("aging") or CONCERN_FOLLOWUPS.get("wrinkle", CONCERN_FOLLOWUPS["acne"])
        return (
            "I'd like to help. Can you tell me a bit more?\n\n"
            "• What exactly is bothering you — breakouts, dryness, redness, or something else?\n"
            "• Where — face, body, or both?\n"
            "• What's your skin type if you know it — oily, dry, combination, or sensitive?\n\n"
            "Then I can suggest ingredients and products that fit."
        )
    return None


def _stringify_ingredient_field(val: Any) -> str:
    """Turn CSV list/tuple/string reprs into readable comma-separated text (fixes ugly [' ', 'Acne', ...] output)."""
    if val is None:
        return ""
    if isinstance(val, float) and val != val:  # NaN
        return ""
    if isinstance(val, (list, tuple, set)):
        bits = [str(x).strip() for x in val if str(x).strip()]
        return ", ".join(bits)
    s = str(val).strip()
    if not s:
        return ""
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                bits = [str(x).strip() for x in parsed if str(x).strip()]
                return ", ".join(bits)
        except (ValueError, SyntaxError):
            pass
    return s


def _format_ingredient(ing: dict) -> str:
    name = ing.get("name", "Unknown")
    what = _stringify_ingredient_field(ing.get("what_is_it", ""))
    what_do = _stringify_ingredient_field(ing.get("what_does_it_do", ""))
    good_for = _stringify_ingredient_field(ing.get("who_is_it_good_for", ""))
    avoid = _stringify_ingredient_field(ing.get("who_should_avoid", ""))
    parts = [name]
    if what:
        parts.append(f"What it is: {str(what)[:400]}{'...' if len(str(what)) > 400 else ''}")
    if what_do:
        parts.append(f"Benefits: {str(what_do)[:400]}{'...' if len(str(what_do)) > 400 else ''}")
    if good_for:
        parts.append(f"Good for: {good_for}")
    if avoid:
        parts.append(f"Avoid if: {avoid}")
    return "\n".join(parts)


def _format_product(prod: dict) -> str:
    name = prod.get("product_name", "Unknown")
    ptype = prod.get("product_type", "")
    price = prod.get("price", "")
    url = prod.get("product_url", "")
    line = f"• {name}"
    if ptype:
        line += f" ({ptype})"
    if price:
        line += f" — {price}"
    if url:
        line += f" — {url}"
    return line


def _is_short_affirmation(text: str) -> bool:
    """True if the message is a short yes/ok style reply."""
    t = text.strip().lower()
    return t in ("yes", "yeah", "yep", "yea", "ok", "okay", "sure", "yup", "please", "go ahead")


CONCERN_WORDS = [
    "dry", "acne", "oil", "oily", "sensitive", "redness", "burn", "burned", "itch",
    "skin", "face", "moistur", "wrinkle", "aging", "pigment", "dull", "flaky", "tight",
]

# Specific problem types — when present we use this message for product fit; otherwise prefer last concern from history
SPECIFIC_CONCERN_WORDS = [
    "dry", "acne", "oil", "oily", "sensitive", "redness", "burn", "burned", "itch",
    "wrinkle", "aging", "pigment", "dull", "flaky", "tight", "breakout", "rash",
]


def _message_has_concern(text: str) -> bool:
    """True if the message clearly describes a skin concern (so we use it for product search)."""
    if not text or len(text.strip()) < 4:
        return False
    t = text.strip().lower()
    return any(c in t for c in CONCERN_WORDS)


def _explicit_ingredient_question(text: str) -> bool:
    """True when the user is asking about a named ingredient or ingredient-led products—not only describing how skin feels."""
    t = (text or "").strip().lower()
    if not t:
        return False
    prefixes = (
        "what is ",
        "what are ",
        "tell me about ",
        "explain ",
        "define ",
        "products with ",
        "products containing ",
        "product with ",
        "containing ",
        "ingredient ",
        "what does ",
    )
    if any(t.startswith(p) for p in prefixes):
        return True
    if "what is " in t or "tell me about " in t or "explain " in t:
        return True
    return False


def _concern_first_message(text: str) -> bool:
    """User is focused on a skin/face problem, not an ingredient encyclopedia question."""
    return _message_has_concern(text) and not _explicit_ingredient_question(text)


def _message_has_specific_concern(text: str) -> bool:
    """True if the message names a specific skin problem (dry, acne, burn, etc.) so we use it for recommendations."""
    if not text or len(text.strip()) < 4:
        return False
    t = text.strip().lower()
    return any(c in t for c in SPECIFIC_CONCERN_WORDS)


def _last_user_concern(conversation_history: list[dict]) -> Optional[str]:
    """Return the last user message that mentions a skin concern, for context after 'yes' / 'ok' or generic 'product for my problem'."""
    for m in reversed(conversation_history):
        if m.get("role") != "user":
            continue
        content = (m.get("content") or "").strip()
        if len(content) > 8 and _message_has_concern(content):
            return content
    return None


def _search_text_for_knowledge(
    user_message: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """
    Build a concise string for ingredient/product search against local data (CSV) and optional MySQL.
    Strips the skin-photo prefix so queries match concerns and ingredients; falls back like retrieval mode.
    """
    history = conversation_history or []
    msg = (user_message or "").strip()
    if not msg:
        return msg
    low = msg.lower()
    marker = "from my skin photo the condition appears to be "
    if marker in low:
        try:
            start = low.index(marker) + len(marker)
            after = msg[start:].strip()
        except ValueError:
            after = msg
        if "." in after:
            cond_part, rest = after.split(".", 1)
            rest = rest.strip()
            merged = f"{cond_part.strip()} {rest}".strip() if rest else cond_part.strip()
            return merged
        return after
    if _message_has_specific_concern(msg):
        return msg
    prev = _last_user_concern(history)
    if prev:
        return prev
    return msg


def _parse_budget(text: str) -> Optional[float]:
    """Parse max price/budget from message (e.g. '15$', 'under 15', 'budget 20', 'max 10 dollars'). Returns None if not found."""
    if not text or not text.strip():
        return None
    # Match numbers with optional $ or "dollars" / "dollar"
    # e.g. 15$, $15, under 15, budget 20, max 10 dollars, under $15
    patterns = [
        r"(?:under|below|max|budget|within)\s*\$?\s*(\d+(?:\.\d+)?)",
        r"\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars?|\$|usd)?",
        r"(\d+(?:\.\d+)?)\s*\$",
    ]
    for pat in patterns:
        m = re.search(pat, text.strip(), re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except (ValueError, IndexError):
                pass
    return None


def _filter_products_by_price(products: list[dict], max_price: float) -> list[dict]:
    """Keep only products with price <= max_price. Price may be string or number."""
    out = []
    for p in products:
        raw = p.get("price")
        if raw is None or raw == "":
            continue
        try:
            price = float(raw) if not isinstance(raw, (int, float)) else raw
            if price <= max_price:
                out.append(p)
        except (ValueError, TypeError):
            continue
    return out


# Terms that must match product name/type text when we have an identified skin concern (budget + recommendations)
_CONCERN_MATCH_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "acne": ("acne", "blemish", "breakout", "bha", "salicylic", "benzoyl", "peroxide", "clarif", "pore"),
    "dry": ("dry", "dehydrat", "hydrat", "moistur", "ceramide", "barrier", "hyaluronic", "glycerin", "humect"),
    "oil": ("oil", "oily", "sebum", "matte", "shine", "t-zone"),
    "oily": ("oil", "oily", "sebum", "matte", "shine", "t-zone"),
    "sensitive": ("sensitive", "calm", "soothe", "gentle", "barrier", "redness", "irritat"),
    "redness": ("redness", "rosacea", "calm", "soothe", "anti-red", "couperose"),
    "wrinkle": ("wrinkle", "line", "firm", "retinol", "peptide", "aging", "anti-age"),
    "aging": ("aging", "wrinkle", "line", "firm", "retinol", "peptide", "mature"),
    "pigment": ("pigment", "bright", "dark spot", "niacinamide", "vitamin c", "uneven", "tone"),
    "dull": ("dull", "bright", "radiance", "glow", "vitamin c"),
    "itch": ("itch", "eczema", "dermatitis", "calm", "oat"),
    "rash": ("rash", "irritat", "calm", "soothe"),
    "burn": ("burn", "soothe", "repair", "barrier"),
    "tight": ("tight", "dry", "hydrat", "moistur"),
    "flaky": ("flaky", "dry", "exfoliat", "hydrat"),
    "breakout": ("breakout", "acne", "blemish", "salicylic"),
}


def _concern_match_terms(search_query: str) -> list[str]:
    """Substrings product name/type should match for relevance to the user's problem."""
    if not search_query or not search_query.strip():
        return []
    t = search_query.lower()
    terms: list[str] = []
    for w in SPECIFIC_CONCERN_WORDS:
        if w in t:
            terms.extend(_CONCERN_MATCH_EXPANSIONS.get(w, (w,)))
    # Extra skin terms (e.g. rosacea, eczema) if present in query
    for extra in ("rosacea", "eczema", "melasma", "hyperpigmentation", "blackhead", "whitehead", "cystic"):
        if extra in t:
            terms.append(extra)
    # Dedupe, preserve order
    seen = set()
    out: list[str] = []
    for x in terms:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _product_blob(p: dict) -> str:
    parts = [
        str(p.get("product_name") or ""),
        str(p.get("product_type") or ""),
        str(p.get("product_url") or ""),
    ]
    return " ".join(parts).lower()


def _filter_products_for_user_problem(products: list[dict], search_query: str) -> list[dict]:
    """Prefer products whose name/type relate to the skin concern; if none match by title, keep search results."""
    terms = _concern_match_terms(search_query)
    if not terms:
        return products
    out = []
    for p in products:
        blob = _product_blob(p)
        if any(term in blob for term in terms):
            out.append(p)
    return out if out else products


# Gemini: role-play as the store receptionist so replies match the online storefront experience.
_LLM_SYSTEM_INSTRUCTION = """You are the professional virtual receptionist for SkinMe, our online skincare store (skinme.store). Sound like a great front-desk person: polished and courteous, genuinely warm, easy to talk to—never stiff or robotic.

Tone: friendly professionalism. You may sprinkle in very light humor now and then—think a quick smile in conversation (a gentle joke about shopping online, routines, or how confusing ingredient names can be). Never use humor about someone's skin, appearance, health, or sensitive concerns. Stay kind and respectful when they describe problems.

Always answer what the customer actually asked. If they ask a question, respond to it directly first; then add helpful context (ingredients, product ideas, or a gentle follow-up) when it fits.

You know skincare and our catalog. Facts about ingredients and products come only from the supplied context. Product names and prices must match the product list exactly when you mention them. Do not invent products, prices, or ingredient facts. If something is missing from the context, say so briefly instead of guessing.

If the user describes a skin or face concern, acknowledge how they feel, give practical guidance in plain language, and avoid sounding like a database. Do not say "matching ingredients", "search results", "retrieved", "aligned", "score", "calculation", "knowledge base", "CSV", or explain how you found the information. Do not present long ingredient lists unless they explicitly asked what an ingredient is or asked for ingredient details.

When they describe a problem and product names are in the context, you may suggest 1–3 products by name and price in a natural way—not as a catalog dump.

Style: Plain, natural prose. No markdown bold (asterisks), no # headings, unless the user asks for formatted lists. Short paragraphs or simple line breaks are fine.

Continue the conversation naturally. When they ask for products within a budget, only use products from the context that fit their stated skin problem (do not suggest unrelated categories).

When the user mentions a skin problem but does NOT ask for products and context is thin, you may ask brief, human follow-up questions.

When the customer sent a skin photo (a separate instruction block may say so), reply like an in-store SkinMe colleague: warm, specific, practical. Never mention machine learning, model training, datasets, image analysis, confidence scores, algorithms, or how the system works technically. Never say you cannot see the image if the instruction says they uploaded one—treat it as a normal consultation and focus on care steps plus products from the reference list.

Language: Reply in English or in Khmer (ភាសាខ្មែរ), matching the customer. If they write in Khmer, reply entirely in Khmer. If they write in English, reply in English. If they switch language, follow them. If they ask you to respond in Khmer (in words or in Khmer), use Khmer. A separate instruction block may tell you which language to use for this turn—follow it. Keep ingredient and product names from the reference as printed when they are Latin; translate your own explanations and advice into the customer language."""


class ChatService:
    """Produces assistant replies from user message and optional history."""

    def __init__(self, repo: Optional[KnowledgeRepository] = None):
        self._repo = repo or KnowledgeRepository()

    def reply_with_retrieval(
        self,
        user_message: str,
        use_database: bool = False,
        conversation_history: Optional[list[dict]] = None,
    ) -> str:
        msg = user_message.strip().lower()
        history = conversation_history or []

        # Short affirmation after we asked a follow-up: use previous user message as context
        if _is_short_affirmation(user_message):
            prev_concern = _last_user_concern(history)
            if prev_concern:
                concern_type = _detect_concern_type(prev_concern)
                use_db = use_database or _is_recommendation_request(prev_concern)
                ing_hits = self._repo.search_ingredients(prev_concern, 6)
                prod_hits = self._repo.search_skinme_products_for_chat_context(
                    prev_concern, ing_hits, product_type=concern_type, top_k=24, use_database=use_db
                )
                prod_hits = _filter_products_for_user_problem(prod_hits, prev_concern)
                max_price = _parse_budget(user_message)
                if max_price is not None:
                    prod_hits = _filter_products_by_price(prod_hits, max_price)[:8]
                    if not prod_hits:
                        prod_hits = self._repo.search_skinme_products_for_chat_context(
                            prev_concern, ing_hits, product_type=concern_type, top_k=20, use_database=use_db
                        )
                        prod_hits = sorted(
                            prod_hits,
                            key=lambda p: float(p.get("price") or 999) if p.get("price") else 999,
                        )[:5]
                else:
                    prod_hits = prod_hits[:5]
                parts = []
                cf = _concern_first_message(prev_concern)
                if prod_hits:
                    if max_price is not None:
                        head = f"Here are a few options within your ${max_price:.0f} budget:\n"
                    elif cf:
                        head = "Thanks for sticking with me. Here are a few products that might suit what you described:\n"
                    else:
                        head = "Here are a few products that might fit:\n"
                    if not cf and ing_hits and max_price is None:
                        parts.append(
                            "A few ingredients people often look for for this kind of concern:\n\n"
                            + "\n\n".join(_format_ingredient(h) for h in ing_hits[:3])
                        )
                    parts.append(head + "\n".join(_format_product(p) for p in prod_hits))
                elif ing_hits and not cf and max_price is None:
                    parts.append("A few ingredients people often look for for this kind of concern:\n\n" + "\n\n".join(_format_ingredient(h) for h in ing_hits[:3]))
                elif ing_hits and cf and not prod_hits and max_price is None:
                    names = [h.get("name") for h in ing_hits[:3] if h.get("name")]
                    if names:
                        return (
                            "That sounds really uncomfortable. I'm not a medical professional, but people often look into "
                            f"ingredients like {', '.join(names)} for concerns like yours. "
                            "If it doesn't settle, a dermatologist can help you figure out what's going on."
                        )
                if parts:
                    if cf and prod_hits:
                        intro = "That sounds rough—skin can be really sensitive to change.\n\n"
                    else:
                        intro = ""
                    return intro + "\n\n".join(parts)
                return (
                    "I’d like to suggest the right products. Can you tell me a bit more? "
                    "For example: Is your skin mostly dry or sensitive? Which area is most affected — face or body?"
                )
            return (
                "Could you tell me a bit more so I can recommend the right products? "
                "For example: What’s your main skin concern — dryness, acne, sensitivity, or something else?"
            )

        # Products with [ingredient]
        for start in ("products with ", "products containing ", "product with ", "containing "):
            if start in msg:
                idx = msg.find(start)
                name = user_message[idx + len(start) :].strip().rstrip("?.!").strip()
                if name:
                    prods = self._repo.get_products_containing_ingredient(name, 6, use_database=use_database)
                    if prods:
                        ing = self._repo.get_ingredient_by_name(name)
                        out = (_format_ingredient(ing) + "\n\n") if ing else ""
                        out += "Products containing it:\n" + "\n".join(_format_product(p) for p in prods)
                        return out
                    ing = self._repo.get_ingredient_by_name(name)
                    if ing:
                        prods = self._repo.get_products_containing_ingredient(
                            ing.get("name", name), 6, use_database=use_database
                        )
                        if prods:
                            return _format_ingredient(ing) + "\n\nProducts:\n" + "\n".join(_format_product(p) for p in prods)
                break

        # What is [ingredient]
        for start in ("what is ", "tell me about ", "ingredient ", "what does ", "explain "):
            if msg.startswith(start):
                name = user_message[len(start) :].strip().rstrip("?.!").strip()
                if name:
                    ing = self._repo.get_ingredient_by_name(name)
                    if ing:
                        out = _format_ingredient(ing)
                        prods = self._repo.get_products_containing_ingredient(
                            ing.get("name", name), 3, use_database=use_database
                        )
                        if prods:
                            out += "\n\nProducts containing it:\n" + "\n".join(_format_product(p) for p in prods)
                        return out
                    hits = self._repo.search_ingredients(name, 3)
                    if hits:
                        return "\n\n".join(_format_ingredient(h) for h in hits)
                break

        # Vague skin problem — ask what type of problem (like a real person would)
        followup = _get_concern_followup(user_message, history)
        if followup:
            return followup

        # Products for concern — ingredients from all datasets; SkinMe products matched to those ingredients + concern
        concern_type = _detect_concern_type(user_message)
        has_product_intent = concern_type or _message_has_concern(user_message) or any(
            w in msg
            for w in (
                "product",
                "recommend",
                "suggest",
                "for my",
                "for dry",
                "for oily",
                "for sensitive",
                "for acne",
                "moisturiser",
                "moisturizer",
                "budget",
                "under",
            )
        )
        if has_product_intent:
            max_price = _parse_budget(user_message)
            search_query = _search_text_for_knowledge(user_message, history)
            concern_type = _detect_concern_type(search_query)
            use_db = use_database or _is_recommendation_request(user_message) or _is_recommendation_request(search_query)
            ing_hits = self._repo.search_ingredients(search_query, 6)
            prod_hits = self._repo.search_skinme_products_for_chat_context(
                search_query, ing_hits, product_type=concern_type, top_k=24, use_database=use_db
            )
            prod_hits = _filter_products_for_user_problem(prod_hits, search_query)
            if max_price is not None:
                within_budget = _filter_products_by_price(prod_hits, max_price)
                if within_budget:
                    prod_hits = within_budget[:8]
                    budget_header = f"Here are products within your ${max_price:.0f} budget:\n"
                else:
                    def _price_val(p):
                        try:
                            return float(p.get("price") or 999)
                        except (ValueError, TypeError):
                            return 999
                    prod_hits = sorted(prod_hits, key=_price_val)[:5]
                    budget_header = f"We don't have products under ${max_price:.0f}. Here are the closest in price:\n"
                if not prod_hits:
                    return (
                        f"I couldn't find anything in our catalog under ${max_price:.0f} for that kind of concern. "
                        "You could try a slightly higher budget, or tell me a bit more about your skin—for example if it's dry, oily, or sensitive."
                    )
                cf_b = _concern_first_message(user_message)
                lead = "I hear you—staying in budget matters.\n\n" if cf_b else ""
                return lead + budget_header + "\n".join(_format_product(p) for p in prod_hits)
            prod_hits = prod_hits[:5]
            cf = _concern_first_message(user_message)
            parts = []
            if prod_hits:
                if ing_hits and not cf:
                    parts.append(
                        "A few ingredients people often look for for this kind of concern:\n\n"
                        + "\n\n".join(_format_ingredient(h) for h in ing_hits[:3])
                    )
                header = (
                    "Here are a few products that might fit what you're describing:\n"
                    if cf
                    else "Some products that might fit:\n"
                )
                parts.append(header + "\n".join(_format_product(p) for p in prod_hits))
            elif ing_hits and not cf:
                parts.append(
                    "A few ingredients people often look for for this kind of concern:\n\n"
                    + "\n\n".join(_format_ingredient(h) for h in ing_hits[:3])
                )
            elif ing_hits and cf and not prod_hits:
                names = [h.get("name") for h in ing_hits[:3] if h.get("name")]
                if names:
                    return (
                        "Thanks for trusting me with that—skin can be really stressful when it acts up. "
                        "I'm not a medical professional, but ingredients like "
                        f"{', '.join(names)} often come up when people talk about this. "
                        "If it keeps bothering you or gets worse, a dermatologist is the right person to check it properly."
                    )
            if parts:
                intro = "I hear you—that can be frustrating.\n\n" if cf and prod_hits else ""
                return intro + "\n\n".join(parts)

        # Greeting — check before ingredient search so "Hi" doesn't match random ingredients
        _khmer_greet = any(p in (user_message or "") for p in ("សួស្តី", "ជំរាបសួរ")) and len((user_message or "").strip()) < 40
        if (any(w in msg for w in ("hi", "hello", "hey")) and len(msg) < 20) or _khmer_greet:
            if _prefer_khmer_reply(user_message, history):
                return (
                    "សួស្តី និងសូមស្វាគមន៍មកកាន់ SkinMe! ខ្ញុំនៅទីនេះដើម្បីជួយអ្នករកផលិតផលថែរស្សីដែលសមនឹងតម្រូវការ។\n\n"
                    "សូមសួរអ្វីក៏បាន ឧទាហរណ៍៖\n"
                    "• [គ្រឿងផ្សំ] ជាអ្វី? (ឧ. «niacinamide ជាអ្វី?»)\n"
                    "• តើអ្នកណែនាំអ្វីសម្រាប់ [បញ្ហាស្បែក]? (ឧ. ស្បែកស្ងួត ឬមុខឡើងពងរមួល)\n"
                    "• តើមានផលិតផលដែលមាន [គ្រឿងផ្សំ] ទេ?\n\n"
                    "សូមប្រាប់ខ្ញុំពីអ្វីដែលអ្នកត្រូវការ ខ្ញុំនឹងជួយអ្នកឱ្យបានល្អបំផុត។"
                )
            return (
                "Hello, and welcome to SkinMe! I’m on reception today—can’t hand you a coffee through the screen, "
                "but I can absolutely help you find the right skincare.\n\n"
                "Ask me anything, for example:\n"
                "• What is [ingredient]? (e.g. \"What is niacinamide?\")\n"
                "• What would you recommend for [concern]? (e.g. dry skin or acne)\n"
                "• Do you have products with [ingredient]?\n\n"
                "Tell me what you need and I’ll do my best."
            )

        # General ingredient search — skip dry "matches" when the user is really talking about skin/face
        ing_hits = self._repo.search_ingredients(user_message, 5)
        if ing_hits:
            if _concern_first_message(user_message):
                names = [h.get("name") for h in ing_hits[:3] if h.get("name")]
                if names:
                    return (
                        "It sounds like your skin has been giving you a hard time. "
                        "I'm not a doctor, but people sometimes read about ingredients "
                        f"like {', '.join(names)} when they're dealing with something like that. "
                        "If you tell me a bit more—or if you want product ideas—I'm happy to help."
                    )
            return "Here's what I found on that:\n\n" + "\n\n".join(_format_ingredient(h) for h in ing_hits)

        if _prefer_khmer_reply(user_message, history):
            return (
                "អូ… ខ្ញុំមិនយល់ច្បាស់ទេ។ សូមនិយាយម្តងទៀតបានទេ? ឧទាហរណ៍៖\n"
                "• niacinamide ជាអ្វី?\n"
                "• តើណែនាំអ្វីសម្រាប់ស្បែកស្ងួត?\n"
                "• តើមានផលិតផលដែលមាន hyaluronic acid ទេ?"
            )
        return (
            "Hmm—I didn’t quite catch that (happens to the best of us). Could you say it another way? For example:\n"
            "• What is niacinamide?\n"
            "• What do you recommend for dry skin?\n"
            "• Do you have products with hyaluronic acid?"
        )

    def reply_with_llm(
        self,
        user_message: str,
        conversation_history: list[dict],
        use_database: bool = False,
        retrieval_query: Optional[str] = None,
        llm_extra_instruction: Optional[str] = None,
    ) -> str:
        api_key = _gemini_api_key()
        if not api_key:
            return self.reply_with_retrieval(
                user_message, use_database=use_database, conversation_history=conversation_history
            )
        try:
            from google import genai as google_genai
            from google.genai import types as genai_types
        except Exception:
            return self.reply_with_retrieval(
                user_message, use_database=use_database, conversation_history=conversation_history
            )

        query_for_search = (retrieval_query or user_message).strip() or user_message
        search_query = _search_text_for_knowledge(query_for_search, conversation_history)
        concern_type = _detect_concern_type(search_query)
        use_db = use_database or _is_recommendation_request(user_message) or _is_recommendation_request(search_query)
        ing_hits = self._repo.search_ingredients(search_query, 6)
        prod_hits = self._repo.search_skinme_products_for_chat_context(
            search_query, ing_hits, product_type=concern_type, top_k=24, use_database=use_db
        )
        prod_hits = _filter_products_for_user_problem(prod_hits, search_query)
        concern_first = _concern_first_message(user_message)
        max_price_llm = _parse_budget(user_message)
        budget_note = ""
        if max_price_llm is not None and prod_hits:
            within = _filter_products_by_price(prod_hits, max_price_llm)
            if within:
                prod_hits = within[:8]
                budget_note = f"Budget: prefer products around ${max_price_llm:.0f} or less."
            else:
                def _pv_llm(p: dict) -> float:
                    try:
                        return float(p.get("price") or 999)
                    except (ValueError, TypeError):
                        return 999.0
                priced = [p for p in prod_hits if p.get("price") not in (None, "")]
                prod_hits = sorted(priced, key=_pv_llm)[:5] if priced else prod_hits[:5]
                budget_note = (
                    f"Budget: nothing in that range; below are the closest alternatives by price."
                )
        else:
            prod_hits = prod_hits[:8]
        context_parts = []
        if ing_hits:
            skip_ingredient_dump = concern_first and bool(prod_hits)
            if skip_ingredient_dump:
                pass
            elif concern_first and not prod_hits:
                names = [h.get("name") for h in ing_hits[:4] if h.get("name")]
                if names:
                    context_parts.append(
                        "Ingredient names that sometimes come up for this kind of concern (for one short sentence in your reply, not a list): "
                        + ", ".join(names)
                    )
            else:
                context_parts.append(
                    "Ingredient notes (for reference):\n"
                    + "\n".join(
                        f"- {h.get('name')}: {_stringify_ingredient_field(h.get('what_is_it', ''))[:200]} | "
                        f"Benefits: {_stringify_ingredient_field(h.get('what_does_it_do', ''))[:200]} | "
                        f"Good for: {_stringify_ingredient_field(h.get('who_is_it_good_for', ''))[:150]}"
                        for h in ing_hits
                    )
                )
        if prod_hits:
            prod_lines = "\n".join(
                f"- {p.get('product_name')} ({p.get('product_type')}) {p.get('price', '')}" for p in prod_hits
            )
            if budget_note:
                prod_lines = budget_note + "\n" + prod_lines
            context_parts.append("Products you may mention (names and prices must match this list exactly):\n" + prod_lines)
        context = "\n\n".join(context_parts) if context_parts else "No specific ingredients or products found."

        system = _LLM_SYSTEM_INSTRUCTION
        history_lines: list[str] = []
        for m in conversation_history[-6:]:
            role = m.get("role")
            c = ((m.get("content") or "")[:500]).strip()
            if not c:
                continue
            if role == "user":
                history_lines.append(f"Customer: {c}")
            elif role == "assistant":
                history_lines.append(f"You (receptionist): {c}")
        blocks: list[str] = []
        if history_lines:
            blocks.append("Recent conversation:\n\n" + "\n\n".join(history_lines))
        blocks.append(
            f"What the user is talking about (carry through the thread if relevant):\n{search_query}\n\n"
            f"Reference information (do not quote source labels to the user):\n{context}\n\n"
            f"User message:\n{user_message}"
        )
        if llm_extra_instruction and llm_extra_instruction.strip():
            blocks.append(llm_extra_instruction.strip())
        if _prefer_khmer_reply(user_message, conversation_history):
            blocks.append(
                "Language for this reply: Khmer (ភាសាខ្មែរ). Write the entire reply in natural, polite Khmer "
                "suited to a professional store receptionist. Keep ingredient and product names from the reference as "
                "printed when they are Latin; translate explanations, advice, and any paraphrased prices into Khmer."
            )
        else:
            blocks.append("Language for this reply: English.")
        user_block = "\n\n---\n\n".join(blocks)

        model_name = (os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash").strip()
        try:
            client = google_genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=model_name,
                contents=user_block,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=640,
                    temperature=0.72 if llm_extra_instruction else (0.7 if concern_first else 0.55),
                ),
            )
            text = (getattr(resp, "text", None) or "").strip()
            if text:
                return _normalize_assistant_plain_text(text)
        except Exception:
            pass
        return self.reply_with_retrieval(
            user_message, use_database=use_database, conversation_history=conversation_history
        )

    def get_reply(
        self,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
        use_llm: bool = True,
        use_database: bool = False,
        retrieval_query: Optional[str] = None,
        llm_extra_instruction: Optional[str] = None,
    ) -> str:
        """
        Prefer Gemini for the final reply when enabled (natural, person-like prose with retrieval context).
        Fall back to canned follow-ups + retrieval only when LLM is off or no API key.

        retrieval_query: optional string used only for ingredient/product lookup (e.g. merge skin hint + user text).
        llm_extra_instruction: optional staff-only guidance appended to the LLM prompt (not shown to user verbatim).
        """
        history = conversation_history or []
        if use_llm and _llm_enabled():
            return self.reply_with_llm(
                user_message,
                history,
                use_database=use_database,
                retrieval_query=retrieval_query,
                llm_extra_instruction=llm_extra_instruction,
            )
        followup = _get_concern_followup(user_message, history)
        if followup:
            return followup
        return self.reply_with_retrieval(
            user_message, use_database=use_database, conversation_history=history
        )
