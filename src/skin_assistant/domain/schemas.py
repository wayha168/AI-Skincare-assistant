"""Domain schemas (DTOs) for API request/response."""
from pydantic import BaseModel, Field
from typing import Optional


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list, max_length=20)
    use_llm: bool = True
    use_database: bool = Field(
        False,
        description="If true, merge MySQL skinme_db products with scraped skinme_products.csv for recommendations (requires MYSQL_* in .env).",
    )
    session_id: Optional[str] = Field(None, max_length=128, description="If set, this turn is saved to DB (skinme_db).")
    user_id: Optional[str] = Field(None, max_length=36, description="Logged-in user ID from FE (e.g. from auth). Stored in DB with chat.")
    user_email: Optional[str] = Field(None, max_length=255, description="Logged-in user email from FE. Stored in DB.")
    user_name: Optional[str] = Field(None, max_length=255, description="Logged-in user name from FE. Stored in DB.")


class ChatResponse(BaseModel):
    reply: str
    options: list[str] = Field(
        default_factory=list,
        description="Quick-reply suggestions for the client UI (chips/buttons).",
    )
    session_id: Optional[str] = None
    admin_connected: Optional[bool] = Field(
        None,
        description="When session_id is set: true if an admin is live on WebSocket for this session.",
    )


class ChatMessageOut(BaseModel):
    role: str
    content: str
    created_at: Optional[str] = None
    is_ai_response: Optional[bool] = None
    sender: Optional[str] = None
    image_analysis: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessageOut]


class ChatSessionSummary(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    session_created_at: Optional[str] = None
    last_message: Optional[str] = None
    last_message_at: Optional[str] = None
    last_message_role: Optional[str] = None
    last_message_sender: Optional[str] = None


class ChatSessionsResponse(BaseModel):
    count: int
    sessions: list[ChatSessionSummary]


class AdminReplyRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    content: str = Field(..., min_length=1, max_length=4000)
    admin_key: Optional[str] = Field(None, description="Required when WS_ADMIN_KEY / ADMIN_API_KEY is set.")
    user_id: Optional[str] = Field(None, max_length=36)
    user_email: Optional[str] = Field(None, max_length=255)
    user_name: Optional[str] = Field(None, max_length=255)


class AdminReplyResponse(BaseModel):
    saved: bool
    delivered_via_websocket: bool
    message_id: str


class ChatWithImageResponse(BaseModel):
    """Response when sending a message with an image for skin analysis."""
    reply: str
    image_analysis: Optional[str] = None  # e.g. "acne (85%)"


# --- Backend integration (e.g. Spring): save to database ---

class ChatLogRequest(BaseModel):
    """Payload for logging a chat turn so backend can persist to DB."""
    session_id: str = Field(..., min_length=1, max_length=128)
    user_id: Optional[str] = Field(None, max_length=128)
    message: str = Field(..., min_length=1, max_length=2000)
    reply: str = Field(..., max_length=10000)
    timestamp: Optional[str] = None  # ISO datetime; backend can set if omitted


class FeedbackRequest(BaseModel):
    """Payload for saving user feedback so backend can persist to DB."""
    session_id: str = Field(..., min_length=1, max_length=128)
    message_id: Optional[str] = Field(None, max_length=128)
    rating: Optional[int] = Field(None, ge=1, le=5)  # 1-5 or use thumbs up/down
    thumbs_up: Optional[bool] = None
    comment: Optional[str] = Field(None, max_length=1000)


class SaveResponse(BaseModel):
    """Response for save endpoints (forwarded to Spring)."""
    saved: bool = True
    message: str = "ok"
    backend_saved: Optional[bool] = None  # True if Spring acknowledged


class IngredientOut(BaseModel):
    name: Optional[str] = None
    scientific_name: Optional[str] = None
    what_is_it: Optional[str] = None
    what_does_it_do: Optional[str] = None
    who_is_it_good_for: Optional[str] = None
    who_should_avoid: Optional[str] = None
    url: Optional[str] = None


class ProductOut(BaseModel):
    product_name: Optional[str] = None
    product_type: Optional[str] = None
    product_url: Optional[str] = None
    price: Optional[str] = None
    product_id: Optional[int] = None
    product_details_url: Optional[str] = None


class SearchIngredientsResponse(BaseModel):
    query: str
    count: int
    ingredients: list[IngredientOut]


class SearchProductsResponse(BaseModel):
    query: str
    count: int
    products: list[ProductOut]
