"""FastAPI route handlers for Skin Assistant API."""
import uuid
from datetime import datetime, timezone
from typing import Optional

import requests
from fastapi import APIRouter, File, Form, Header, HTTPException, Query, UploadFile

from skin_assistant.api.admin_auth import is_admin_authenticated
from skin_assistant.api.ws_manager import ws_manager
from skin_assistant.config import get_settings
from skin_assistant.domain.schemas import (
    AdminReplyRequest,
    AdminReplyResponse,
    ChatHistoryResponse,
    ChatMessageOut,
    ChatRequest,
    ChatResponse,
    ChatSessionsResponse,
    ChatSessionSummary,
    ChatWithImageResponse,
    ChatLogRequest,
    FeedbackRequest,
    SaveResponse,
    IngredientOut,
    ProductOut,
    SearchIngredientsResponse,
    SearchProductsResponse,
)
from skin_assistant.infrastructure import KnowledgeRepository, ChatRepository
from skin_assistant.services import ChatService
from skin_assistant.services.chat_options import get_suggested_options

from skin_assistant.api.ws_routes import router as ws_router

router = APIRouter(prefix="/v1", tags=["skin-assistant"])
router.include_router(ws_router)

_repo = KnowledgeRepository()

_chat = ChatService(repo=_repo)
_chat_repo = ChatRepository()


def _require_admin(admin_key: Optional[str]) -> None:
    if not is_admin_authenticated(admin_key):
        raise HTTPException(status_code=401, detail="Invalid or missing admin key")


def _admin_key_from(
    query_key: Optional[str],
    header_key: Optional[str],
    body_key: Optional[str] = None,
) -> Optional[str]:
    return (body_key or header_key or query_key or "").strip() or None


def _condition_from_image_analysis(image_analysis: str) -> str:
    """Strip trailing '(NN%)' confidence suffix from classifier label."""
    s = (image_analysis or "").strip()
    if not s:
        return ""
    if " (" in s and s.endswith(")"):
        return s.rsplit(" (", 1)[0].strip()
    return s


def _forward_to_backend(path: str, payload: dict) -> bool:
    """POST payload to Spring (or other) backend for saving to database. Returns True if backend responded 2xx."""
    base = get_settings().backend_url
    if not base:
        return False
    url = f"{base}{path}" if path.startswith("/") else f"{base}/{path}"
    try:
        r = requests.post(url, json=payload, timeout=5)
        return 200 <= r.status_code < 300
    except Exception:
        return False


def _to_ingredient_out(d: dict) -> IngredientOut:
    return IngredientOut(
        name=d.get("name"),
        scientific_name=d.get("scientific_name"),
        what_is_it=d.get("what_is_it"),
        what_does_it_do=d.get("what_does_it_do"),
        who_is_it_good_for=d.get("who_is_it_good_for"),
        who_should_avoid=d.get("who_should_avoid"),
        url=d.get("url"),
    )


def _to_product_out(d: dict) -> ProductOut:
    product_id = d.get("product_id")
    product_details_url = None
    if product_id:
        product_details_url = f"https://skinme.store/product_details?productId={product_id}"
    
    return ProductOut(
        product_name=d.get("product_name"),
        product_type=d.get("product_type"),
        product_url=d.get("product_url"),
        price=d.get("price"),
        product_id=product_id,
        product_details_url=product_details_url,
    )


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Send a message and get the assistant reply. Set use_database=true to merge MySQL products with scraped skinme_products.csv.
    If session_id is set and MySQL is configured, the turn is saved to chat_ai. When user is logged in, send user_id (and optionally user_email, user_name) to store in DB."""
    if req.session_id and _chat_repo.is_available():
        db_history = _chat_repo.get_history(req.session_id, limit=20)
        history = [{"role": r["role"], "content": r["content"] or ""} for r in db_history]
    else:
        history = [{"role": m.role, "content": m.content} for m in req.history]
    reply = _chat.get_reply(
        req.message,
        conversation_history=history,
        use_llm=req.use_llm,
        use_database=req.use_database,
    )
    if req.session_id and _chat_repo.is_available():
        _chat_repo.save_message(
            req.session_id, "user", req.message,
            user_id=req.user_id, user_email=req.user_email, user_name=req.user_name,
        )
        _chat_repo.save_message(
            req.session_id, "assistant", reply,
            user_id=req.user_id, user_email=req.user_email, user_name=req.user_name,
            is_ai_response=True,
        )
    options = get_suggested_options(req.message, reply)
    admin_connected = (
        ws_manager.is_admin_connected(req.session_id) if req.session_id else None
    )
    return ChatResponse(
        reply=reply,
        options=options,
        session_id=req.session_id,
        admin_connected=admin_connected,
    )


@router.get("/chat/sessions", response_model=ChatSessionsResponse)
def list_chat_sessions(
    limit: int = Query(50, ge=1, le=200),
    admin_key: Optional[str] = Query(None, description="Admin API key (or use X-Admin-Key header)."),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> ChatSessionsResponse:
    """List recent chat sessions for admin dashboards (requires admin key when configured)."""
    _require_admin(_admin_key_from(admin_key, x_admin_key))
    if not _chat_repo.is_available():
        raise HTTPException(status_code=503, detail="Chat database is not configured (set MYSQL_* in .env).")
    rows = _chat_repo.list_sessions(limit=limit)
    sessions = [
        ChatSessionSummary(
            session_id=r.get("session_id") or "",
            user_id=r.get("user_id"),
            user_email=r.get("user_email"),
            user_name=r.get("user_name"),
            session_created_at=str(r["session_created_at"]) if r.get("session_created_at") else None,
            last_message=r.get("last_message"),
            last_message_at=str(r["last_message_at"]) if r.get("last_message_at") else None,
            last_message_role=r.get("last_message_role"),
            last_message_sender=r.get("last_message_sender"),
        )
        for r in rows
    ]
    return ChatSessionsResponse(count=len(sessions), sessions=sessions)


@router.get("/chat/sessions/{session_id}/history", response_model=ChatHistoryResponse)
def get_chat_history(
    session_id: str,
    limit: int = Query(100, ge=1, le=500),
    admin_key: Optional[str] = Query(None),
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> ChatHistoryResponse:
    """Return full chat history for a session (admin / backend integration)."""
    _require_admin(_admin_key_from(admin_key, x_admin_key))
    if not _chat_repo.is_available():
        raise HTTPException(status_code=503, detail="Chat database is not configured (set MYSQL_* in .env).")
    rows = _chat_repo.get_history(session_id, limit=limit)
    messages = [
        ChatMessageOut(
            role=r.get("role") or "user",
            content=r.get("content") or "",
            created_at=str(r["created_at"]) if r.get("created_at") else None,
            is_ai_response=bool(r.get("is_ai_response")) if r.get("is_ai_response") is not None else None,
            sender=r.get("sender"),
            image_analysis=r.get("image_analysis"),
        )
        for r in rows
    ]
    return ChatHistoryResponse(session_id=session_id, messages=messages)


@router.post("/chat/admin-reply", response_model=AdminReplyResponse)
async def admin_reply(
    req: AdminReplyRequest,
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> AdminReplyResponse:
    """
    Admin sends a reply to a user session (for Spring/backend integration).
    Saves to MySQL and pushes to the user WebSocket when connected.
    """
    _require_admin(_admin_key_from(None, x_admin_key, req.admin_key))
    message_id = str(uuid.uuid4())
    saved = False
    if _chat_repo.is_available():
        saved = _chat_repo.save_message(
            req.session_id,
            "assistant",
            req.content,
            user_id=req.user_id,
            user_email=req.user_email,
            user_name=req.user_name,
            from_admin=True,
        )
    delivered = await ws_manager.send_json(
        req.session_id,
        "user",
        {
            "type": "message",
            "role": "assistant",
            "content": req.content,
            "message_id": message_id,
            "sender": "admin",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return AdminReplyResponse(
        saved=saved,
        delivered_via_websocket=delivered,
        message_id=message_id,
    )


@router.post("/chat/with-image", response_model=ChatWithImageResponse)
async def chat_with_image(
    message: str = Form("", max_length=2000),
    session_id: Optional[str] = Form(None, max_length=128),
    user_id: Optional[str] = Form(None, max_length=36),
    user_email: Optional[str] = Form(None, max_length=255),
    user_name: Optional[str] = Form(None, max_length=255),
    use_llm: bool = Form(True),
    use_database: bool = Form(False),
    training_label: Optional[str] = Form(
        None,
        description="Optional condition label to save this image directly in labeled training folder.",
    ),
    image: UploadFile = File(...),
) -> ChatWithImageResponse:
    """Send a message with a skin image; we analyze the image and reply with recommendations.
    Message can be empty (we'll ask what to recommend). Turn is saved to DB if session_id is set and MySQL configured."""
    image_analysis = None
    saved_training_path = None
    user_text = (message or "").strip() or "What do you recommend for my skin?"
    try:
        from skin_assistant.models.skin_condition_trainer import (
            predict_skin_condition_from_image,
            save_uploaded_skin_image_for_training,
        )
    except ImportError:
        predict_skin_condition_from_image = None
        save_uploaded_skin_image_for_training = None
    try:
        contents = await image.read()
        if save_uploaded_skin_image_for_training:
            saved = save_uploaded_skin_image_for_training(
                image_bytes=contents,
                original_filename=image.filename or "",
                user_message=user_text,
                session_id=session_id,
                condition_label=training_label,
            )
            if saved:
                saved_training_path = str(saved)
        if predict_skin_condition_from_image:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            condition, conf = predict_skin_condition_from_image(img)
            if condition:
                image_analysis = f"{condition} ({conf:.0%})"
    except Exception:
        pass
    finally:
        await image.close()
    if session_id and _chat_repo.is_available():
        db_history = _chat_repo.get_history(session_id, limit=20)
        image_history = [{"role": r["role"], "content": r["content"] or ""} for r in db_history]
    else:
        image_history = []

    cond_label = _condition_from_image_analysis(image_analysis) if image_analysis else ""
    if cond_label:
        retrieval_query = f"{cond_label} skin concern gentle care moisturizer serum cleanser {user_text}"
        llm_extra = (
            "The customer uploaded a skin photo. Internal screening suggests their concern may lean toward: "
            f"{cond_label}. Reply as a warm SkinMe store colleague: acknowledge their message and photo naturally, "
            "offer brief practical care tips in plain language (no diagnosis), then recommend a few products from the "
            "reference list by exact name and price. Do not mention models, training, datasets, percentages, or "
            "\"image analysis\"."
        )
    else:
        retrieval_query = (
            f"gentle hydrating calming skin barrier cleanser serum moisturizer sensitive skin {user_text}"
        )
        llm_extra = (
            "The customer uploaded a skin photo along with their message. Reply as a warm SkinMe store colleague: "
            "thank them for the photo, respond to what they wrote, ask one short caring question if their concern is "
            "still unclear, and suggest a few suitable products from the reference list (hydration, barrier support, "
            "gentle cleansing) using exact names and prices. Do not mention AI, machine learning, model training, "
            "datasets, or technical screening."
        )

    message_for_model = user_text
    if not use_llm:
        if cond_label:
            message_for_model = f"My skin may relate to {cond_label}. {user_text}".strip()
        else:
            message_for_model = f"I shared a skin photo. {user_text}".strip()

    reply = _chat.get_reply(
        message_for_model,
        conversation_history=image_history,
        use_llm=use_llm,
        use_database=use_database,
        retrieval_query=retrieval_query,
        llm_extra_instruction=llm_extra,
    )
    
    # When image analysis detects a condition, ensure products are shown with clickable links
    if cond_label and use_llm:
        # Search for products based on the detected condition
        prod_hits = _repo.search_products_by_concern(
            cond_label, product_type=None, top_k=5, use_database=use_database
        )
        if prod_hits:
            product_section = "\n\n**Recommended products for your " + cond_label.lower() + " skin:**\n"
            for p in prod_hits[:5]:
                product_name = p.get('product_name', 'Unknown')
                product_type = p.get('product_type', '')
                price = p.get('price', '')
                product_id = p.get('product_id') or p.get('id')
                
                line = f"• {product_name}"
                if product_type:
                    line += f" ({product_type})"
                if price:
                    line += f" — {price}"
                if product_id:
                    line += f" — [View Details](https://skinme.store/product_details?productId={product_id})"
                
                product_section += line + "\n"
            
            reply += product_section
    
    if session_id and _chat_repo.is_available():
        user_content = (message or "Analyze my skin").strip() or "What do you recommend?"
        _chat_repo.save_message(
            session_id, "user",
            user_content
            + (f" [Image analyzed: {image_analysis}]" if image_analysis else "")
            + (f" [Saved for training: {saved_training_path}]" if saved_training_path else ""),
            image_analysis=image_analysis,
            image_path=saved_training_path,
            image_filename=image.filename or None,
            training_label=training_label,
            user_id=user_id, user_email=user_email, user_name=user_name,
        )
        _chat_repo.save_message(
            session_id, "assistant", reply,
            user_id=user_id, user_email=user_email, user_name=user_name,
        )
    return ChatWithImageResponse(reply=reply, image_analysis=image_analysis)


@router.get("/ingredients/search", response_model=SearchIngredientsResponse)
def search_ingredients(
    q: str = Query(..., min_length=1, max_length=200),
    top_k: int = Query(5, ge=1, le=20),
) -> SearchIngredientsResponse:
    """Search ingredients by name or description."""
    results = _repo.search_ingredients(q, top_k=top_k)
    return SearchIngredientsResponse(
        query=q,
        count=len(results),
        ingredients=[_to_ingredient_out(r) for r in results],
    )


@router.get("/ingredients/{name}", response_model=IngredientOut)
def get_ingredient(name: str) -> IngredientOut:
    """Get a single ingredient by name."""
    ing = _repo.get_ingredient_by_name(name)
    if not ing:
        raise HTTPException(status_code=404, detail="Ingredient not found")
    return _to_ingredient_out(ing)


@router.get("/products", response_model=SearchProductsResponse)
def search_products(
    concern: Optional[str] = Query(None, max_length=200),
    product_type: Optional[str] = Query(None, max_length=100),
    ingredient: Optional[str] = Query(None, max_length=200),
    top_k: int = Query(5, ge=1, le=20),
    use_database: bool = Query(
        False,
        description="If true, merge results from MySQL skinme_db with scraped skinme_products.csv.",
    ),
) -> SearchProductsResponse:
    """
    Search products by concern or by ingredient mention.
    Uses skinme_products.csv; adds MySQL when use_database=true and MYSQL_* is configured.
    """
    if ingredient:
        results = _repo.get_products_containing_ingredient(
            ingredient, top_k=top_k, use_database=use_database
        )
        q = ingredient
    elif concern:
        results = _repo.search_products_by_concern(
            concern, product_type=product_type, top_k=top_k, use_database=use_database
        )
        q = concern
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'concern' or 'ingredient' query parameter.",
        )
    return SearchProductsResponse(query=q, count=len(results), products=[_to_product_out(r) for r in results])


@router.get("/intent")
def predict_intent(q: str = Query(..., min_length=1, max_length=500)) -> dict:
    """Predict intent of a message (e.g. greeting, ingredient_info). Requires a trained model in models/artifacts."""
    try:
        from skin_assistant.models import IntentPredictor
        predictor = IntentPredictor()
        intent = predictor.predict(q)
        proba = predictor.predict_proba(q)
        return {"query": q, "intent": intent, "probabilities": proba}
    except Exception:
        return {"query": q, "intent": "other", "probabilities": {}}


# --- Backend integration (Spring): save to database ---

@router.post("/chat/log", response_model=SaveResponse)
def chat_log(req: ChatLogRequest) -> SaveResponse:
    """
    Log a chat turn for persistence in your backend database.
    Spring can expose POST /api/v1/chat/log to receive this payload and save to DB.
    If SPRING_BACKEND_URL is set, the payload is also forwarded to {url}/api/v1/chat/log.
    """
    payload = req.model_dump()
    backend_saved = _forward_to_backend("/api/v1/chat/log", payload)
    return SaveResponse(saved=True, backend_saved=backend_saved if get_settings().backend_url else None)


@router.post("/feedback", response_model=SaveResponse)
def feedback(req: FeedbackRequest) -> SaveResponse:
    """
    Save user feedback for persistence in your backend database.
    Spring can expose POST /api/v1/feedback to receive this payload and save to DB.
    If SPRING_BACKEND_URL is set, the payload is also forwarded to {url}/api/v1/feedback.
    """
    payload = req.model_dump()
    backend_saved = _forward_to_backend("/api/v1/feedback", payload)
    return SaveResponse(saved=True, backend_saved=backend_saved if get_settings().backend_url else None)


@router.get("/routes")
def list_routes() -> dict:
    """
    List route paths for backend (e.g. Spring) integration.
    Use these paths when calling this API or when implementing endpoints that receive forwarded payloads.
    """
    base = "/v1"
    return {
        "base": base,
        "chat": {
            "post_chat": f"POST {base}/chat (body: message, session_id?, user_id?, user_email?, user_name?; response includes options[])",
            "post_chat_with_image": f"POST {base}/chat/with-image (multipart: message, session_id?, user_id?, user_email?, user_name?, image)",
            "post_chat_log": f"POST {base}/chat/log",
            "get_chat_sessions": f"GET {base}/chat/sessions?admin_key=... (admin: list sessions)",
            "get_chat_history": f"GET {base}/chat/sessions/{{session_id}}/history?admin_key=...",
            "post_admin_reply": f"POST {base}/chat/admin-reply (admin: reply to user session)",
            "websocket_chat": f"WS {base}/ws/chat/{{session_id}}?role=user|admin&admin_key=...",
        },
        "feedback": {"post_feedback": f"POST {base}/feedback"},
        "ingredients": {
            "search": f"GET {base}/ingredients/search?q=...",
            "get_by_name": f"GET {base}/ingredients/{{name}}",
        },
        "products": {"search": f"GET {base}/products?concern=...|ingredient=..."},
        "intent": {"predict": f"GET {base}/intent?q=..."},
        "health": f"GET {base}/health",
    }


@router.get("/health")
def health() -> dict:
    """Health check for load balancers and monitoring."""
    return {"status": "ok", "service": "skin-assistant"}
