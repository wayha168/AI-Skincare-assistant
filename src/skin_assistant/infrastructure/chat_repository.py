"""
Persist chat messages to MySQL (skinme_db). Uses same MYSQL_* env as SkinMeDBClient.
Creates chat_sessions and chat_ai tables. Generates a user UUID per session if not provided (same session = same user).
"""
import uuid
from typing import List, Optional

from skin_assistant.config import get_settings


def _get_connection():
    """Return a DB connection or None if MySQL not configured."""
    s = get_settings()
    if not s.use_mysql_db:
        return None
    try:
        import pymysql
    except ImportError:
        return None
    try:
        return pymysql.connect(
            host=s.mysql_host,
            port=getattr(s, "mysql_port", 3306),
            user=s.mysql_user,
            password=s.mysql_password,
            database=s.mysql_database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=5,
        )
    except Exception:
        return None


def _ensure_tables(conn) -> bool:
    """Create chat_sessions and chat_ai if not exist. Returns True on success."""
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    session_id VARCHAR(128) NOT NULL UNIQUE,
                    user_id VARCHAR(36) NULL,
                    user_email VARCHAR(255) NULL,
                    user_name VARCHAR(255) NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_session (session_id),
                    INDEX idx_user (user_id)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_ai (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(36) NOT NULL,
                    session_id VARCHAR(128) NOT NULL,
                    role VARCHAR(32) NOT NULL,
                    content TEXT NOT NULL,
                    image_analysis VARCHAR(255) NULL,
                    image_path VARCHAR(1024) NULL,
                    image_filename VARCHAR(255) NULL,
                    training_label VARCHAR(128) NULL,
                    is_ai_response TINYINT(1) NOT NULL DEFAULT 0,
                    sender VARCHAR(255) NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user_session (user_id, session_id),
                    INDEX idx_session_created (session_id, created_at),
                    INDEX idx_is_ai (is_ai_response)
                )
            """)
            try:
                cur.execute("ALTER TABLE chat_sessions ADD COLUMN user_email VARCHAR(255) NULL")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE chat_sessions ADD COLUMN user_name VARCHAR(255) NULL")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE chat_ai ADD COLUMN is_ai_response TINYINT(1) NOT NULL DEFAULT 0")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE chat_ai ADD COLUMN sender VARCHAR(255) NULL")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE chat_ai ADD COLUMN image_path VARCHAR(1024) NULL")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE chat_ai ADD COLUMN image_filename VARCHAR(255) NULL")
            except Exception:
                pass
            try:
                cur.execute("ALTER TABLE chat_ai ADD COLUMN training_label VARCHAR(128) NULL")
            except Exception:
                pass
        conn.commit()
        return True
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return False


class ChatRepository:
    """Save and load chat messages in MySQL (skinme_db)."""

    def __init__(self):
        self._settings = get_settings()

    def is_available(self) -> bool:
        return bool(self._settings.use_mysql_db)

    def _conn(self):
        return _get_connection()

    def ensure_tables(self) -> bool:
        conn = self._conn()
        if conn is None:
            return False
        try:
            return _ensure_tables(conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def ensure_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        user_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Insert session if not exists. If user_id is not provided, generate a UUID so same session = same user.
        When user is logged in (FE sends user_id, user_email, user_name), store them in DB.
        Returns the user_id for this session (existing or newly generated).
        """
        conn = self._conn()
        if conn is None:
            return None
        try:
            _ensure_tables(conn)
            sid = session_id[:128]
            uid = (user_id or "").strip() or None
            email = (user_email or "").strip()[:255] or None
            name = (user_name or "").strip()[:255] or None
            with conn.cursor() as cur:
                cur.execute("SELECT user_id, user_email, user_name FROM chat_sessions WHERE session_id = %s", (sid,))
                row = cur.fetchone()
                if row and row.get("user_id"):
                    if uid or email or name:
                        cur.execute(
                            "UPDATE chat_sessions SET user_id = COALESCE(%s, user_id), user_email = COALESCE(%s, user_email), user_name = COALESCE(%s, user_name) WHERE session_id = %s",
                            (uid[:36] if uid else None, email, name, sid),
                        )
                    conn.commit()
                    return row["user_id"]
                if not uid:
                    uid = str(uuid.uuid4())
                cur.execute(
                    "INSERT IGNORE INTO chat_sessions (session_id, user_id, user_email, user_name) VALUES (%s, %s, %s, %s)",
                    (sid, uid[:36], email, name),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        "UPDATE chat_sessions SET user_id = COALESCE(%s, user_id), user_email = COALESCE(%s, user_email), user_name = COALESCE(%s, user_name) WHERE session_id = %s AND (user_id IS NULL OR user_id = '')",
                        (uid[:36], email, name, sid),
                    )
                conn.commit()
            return uid[:36]
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            return None
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        image_analysis: Optional[str] = None,
        image_path: Optional[str] = None,
        image_filename: Optional[str] = None,
        training_label: Optional[str] = None,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        user_name: Optional[str] = None,
        *,
        from_admin: bool = False,
        is_ai_response: Optional[bool] = None,
    ) -> bool:
        """
        Append one chat message to chat_ai. Creates session if needed.
        Sets is_ai_response (1=AI, 0=user/admin) and sender (e.g. 'assistant', user email, or 'user').
        """
        if not self.is_available():
            return False
        uid = self.ensure_session(session_id, user_id=user_id, user_email=user_email, user_name=user_name)
        if not uid:
            return False
        role_lower = (role or "").strip().lower()
        if from_admin:
            is_ai = 0
            sender = "admin"
        elif is_ai_response is not None:
            is_ai = 1 if is_ai_response else 0
            sender = "assistant" if role_lower == "assistant" else (
                (user_email or "").strip()[:255] or (user_id or "").strip()[:255] or "user"
            )
        elif role_lower == "assistant":
            is_ai = 1
            sender = "assistant"
        else:
            is_ai = 0
            sender = (user_email or "").strip()[:255] or (user_id or "").strip()[:255] or "user"
        conn = self._conn()
        if conn is None:
            return False
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_ai (user_id, session_id, role, content, image_analysis, image_path, image_filename, training_label, is_ai_response, sender) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        uid[:36],
                        session_id[:128],
                        role[:32],
                        content[:65535],
                        (image_analysis or "")[:255] or None,
                        (image_path or "")[:1024] or None,
                        (image_filename or "")[:255] or None,
                        (training_label or "")[:128] or None,
                        is_ai,
                        sender[:255] if sender else None,
                    ),
                )
            conn.commit()
            return True
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            return False
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def get_history(self, session_id: str, limit: int = 100) -> List[dict]:
        """Return messages for session from chat_ai, newest last. Each dict: role, content, image_analysis, created_at (and is_ai_response, sender if present)."""
        conn = self._conn()
        if conn is None:
            return []
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, content, image_analysis, image_path, image_filename, training_label, is_ai_response, sender, created_at FROM chat_ai WHERE session_id = %s ORDER BY id ASC LIMIT %s",
                    (session_id[:128], limit),
                )
                rows = cur.fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def list_sessions(self, limit: int = 50) -> List[dict]:
        """Recent chat sessions for admin dashboards (newest activity first)."""
        conn = self._conn()
        if conn is None:
            return []
        try:
            _ensure_tables(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        cs.session_id,
                        cs.user_id,
                        cs.user_email,
                        cs.user_name,
                        cs.created_at AS session_created_at,
                        last_msg.content AS last_message,
                        last_msg.created_at AS last_message_at,
                        last_msg.role AS last_message_role,
                        last_msg.sender AS last_message_sender
                    FROM chat_sessions cs
                    LEFT JOIN chat_ai last_msg ON last_msg.id = (
                        SELECT MAX(id) FROM chat_ai WHERE session_id = cs.session_id
                    )
                    ORDER BY last_msg.created_at DESC, cs.created_at DESC
                    LIMIT %s
                    """,
                    (max(1, min(limit, 200)),),
                )
                rows = cur.fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
        finally:
            try:
                conn.close()
            except Exception:
                pass
