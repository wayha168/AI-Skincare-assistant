from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from fastapi import WebSocket


@dataclass
class SessionSockets:
    user: Optional[WebSocket] = None
    admin: Optional[WebSocket] = None


class WebSocketSessionManager:
    """In-memory mapping of session_id -> user_ws/admin_ws.

    Note: This is process-local. For multi-worker deployment you must use a shared
    backend (Redis pubsub, etc.). For a practicum / single instance, this is fine.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionSockets] = {}

    def _get_or_create(self, session_id: str) -> SessionSockets:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionSockets()
        return self._sessions[session_id]

    async def connect_user(self, session_id: str, websocket: WebSocket) -> None:
        s = self._get_or_create(session_id)
        s.user = websocket

    async def connect_admin(self, session_id: str, websocket: WebSocket) -> None:
        s = self._get_or_create(session_id)
        s.admin = websocket

    def disconnect(self, session_id: str, role: str) -> None:
        if session_id not in self._sessions:
            return
        s = self._sessions[session_id]
        if role == "admin":
            s.admin = None
        else:
            s.user = None

    def get_sockets(self, session_id: str) -> Tuple[Optional[WebSocket], Optional[WebSocket]]:
        s = self._sessions.get(session_id)
        if not s:
            return None, None
        return s.user, s.admin

    def is_admin_connected(self, session_id: str) -> bool:
        _, admin_ws = self.get_sockets(session_id)
        return admin_ws is not None

    async def send_json(self, session_id: str, role: str, payload: dict[str, Any]) -> bool:
        """Push a JSON payload to user or admin socket if connected."""
        user_ws, admin_ws = self.get_sockets(session_id)
        target = admin_ws if role == "admin" else user_ws
        if not target:
            return False
        await target.send_json(payload)
        return True


# Shared process-local manager (used by WebSocket routes and REST admin reply).
ws_manager = WebSocketSessionManager()

