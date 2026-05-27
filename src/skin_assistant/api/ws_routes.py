from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from skin_assistant.infrastructure import ChatRepository, KnowledgeRepository
from skin_assistant.services import ChatService
from skin_assistant.services.chat_options import get_suggested_options

from .admin_auth import is_admin_authenticated
from .ws_manager import ws_manager

router = APIRouter()

_chat_repo = ChatRepository()
_chat_service = ChatService(repo=KnowledgeRepository())


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _history_payload(session_id: str, messages: list[dict]) -> dict[str, Any]:
    return {
        "type": "history",
        "session_id": session_id,
        "messages": [
            {
                "role": m.get("role"),
                "content": m.get("content"),
                "created_at": m.get("created_at"),
                "is_ai_response": m.get("is_ai_response"),
                "sender": m.get("sender"),
            }
            for m in messages
        ],
    }


def _message_payload(
    role: str,
    content: str,
    *,
    options: Optional[list[str]] = None,
    sender: Optional[str] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "message",
        "role": role,
        "content": content,
        "message_id": str(uuid.uuid4()),
        "created_at": _utc_now_iso(),
    }
    if sender:
        payload["sender"] = sender
    if options:
        payload["options"] = options
    return payload


@router.websocket("/ws/chat/{session_id}")
async def chat_ws(websocket: WebSocket, session_id: str):
    """Realtime chat WebSocket for users and admins.

    URL: ``ws://host/v1/ws/chat/{session_id}?role=user|admin&admin_key=...``

    Client → server::
        {"type": "message", "content": "...", "user_id"?, "user_email"?, "user_name"?}

    Server → client::
        {"type": "history", "session_id": "...", "messages": [...]}
        {"type": "message", "role": "user|assistant", "content": "...", "options"?: [...], ...}
        {"type": "error", "message": "..."}

  - User messages go to a connected admin (human reply). Otherwise the AI replies with quick options.
  - Admin messages are delivered to the user and stored with sender ``admin``.
    """

    session_id = session_id.strip() or str(uuid.uuid4())
    role = websocket.query_params.get("role", "user").strip().lower()
    admin_key = websocket.query_params.get("admin_key")

    if role not in ("user", "admin"):
        role = "user"

    if role == "admin" and not is_admin_authenticated(admin_key):
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Unauthorized admin"})
        await websocket.close()
        return

    await websocket.accept()

    try:
        if role == "admin":
            await ws_manager.connect_admin(session_id, websocket)
        else:
            await ws_manager.connect_user(session_id, websocket)
            _chat_repo.ensure_session(session_id)

        history = _chat_repo.get_history(session_id, limit=50) if _chat_repo.is_available() else []
        await websocket.send_json(_history_payload(session_id, history))

        while True:
            data = await websocket.receive_json()
            msg_type = (data.get("type") or "message").strip().lower()
            if msg_type != "message":
                continue

            content = (data.get("content") or "").strip()
            if not content:
                continue

            user_id = data.get("user_id")
            user_email = data.get("user_email")
            user_name = data.get("user_name")

            if role == "admin":
                payload = _message_payload("assistant", content, sender="admin")
                await ws_manager.send_json(session_id, "user", payload)
                if _chat_repo.is_available():
                    _chat_repo.save_message(
                        session_id,
                        "assistant",
                        content,
                        user_id=user_id,
                        user_email=user_email,
                        user_name=user_name,
                        from_admin=True,
                    )
                # Forward to backend for persistence (Spring)
                try:
                    from skin_assistant.config import get_settings
                    from skin_assistant.api.routes import _forward_to_backend
                    _forward_to_backend(
                        "/api/v1/chat/log",
                        {
                            "session_id": session_id,
                            "role": "assistant",
                            "content": content,
                            "sender": "admin",
                            "user_id": user_id,
                            "user_email": user_email,
                            "user_name": user_name,
                            "from_admin": True,
                        },
                    )
                except Exception:
                    pass
                continue

            # User message
            user_payload = _message_payload("user", content, sender="user")
            if ws_manager.is_admin_connected(session_id):
                await ws_manager.send_json(session_id, "admin", user_payload)
                if _chat_repo.is_available():
                    _chat_repo.save_message(
                        session_id,
                        "user",
                        content,
                        user_id=user_id,
                        user_email=user_email,
                        user_name=user_name,
                    )
                continue

            conversation_history = []
            if _chat_repo.is_available():
                db_history = _chat_repo.get_history(session_id, limit=20)
                conversation_history = [
                    {"role": r.get("role"), "content": r.get("content") or ""}
                    for r in db_history
                ]

            reply = _chat_service.get_reply(
                content,
                conversation_history=conversation_history,
                use_llm=True,
                use_database=False,
            )
            options = get_suggested_options(content, reply)
            assistant_payload = _message_payload(
                "assistant",
                reply,
                options=options,
                sender="assistant",
            )

            if _chat_repo.is_available():
                _chat_repo.save_message(session_id, "user", content, user_id=user_id, user_email=user_email, user_name=user_name)
                _chat_repo.save_message(
                    session_id,
                    "assistant",
                    reply,
                    user_id=user_id,
                    user_email=user_email,
                    user_name=user_name,
                    is_ai_response=True,
                )

            # Forward to backend for persistence (Spring)
            try:
                from skin_assistant.api.routes import _forward_to_backend
                _forward_to_backend(
                    "/api/v1/chat/log",
                    {
                        "session_id": session_id,
                        "role": "user",
                        "content": content,
                        "sender": "user",
                        "user_id": user_id,
                        "user_email": user_email,
                        "user_name": user_name,
                        "is_ai_response": False,
                    },
                )
                _forward_to_backend(
                    "/api/v1/chat/log",
                    {
                        "session_id": session_id,
                        "role": "assistant",
                        "content": reply,
                        "sender": "assistant",
                        "user_id": user_id,
                        "user_email": user_email,
                        "user_name": user_name,
                        "is_ai_response": True,
                    },
                )
            except Exception:
                pass

            await ws_manager.send_json(session_id, "user", assistant_payload)

    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(session_id, role)
