import json
import uuid
import logging
import os
import re
import time
from typing import List, Union, Optional, Any, Dict
from urllib.parse import urlparse, quote
from fastapi import Request
from pydantic import BaseModel, Field
from supabase import create_client, Client

logger = logging.getLogger(__name__)

# --- 設定 ---
STORAGE_BUCKET_NAME = os.getenv("SUPABASE_STORAGE_BUCKET", "images")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    error_msg = "Critical Error: SUPABASE_URL or SUPABASE_SERVICE_KEY is missing."
    logger.critical(error_msg)
    raise ValueError(error_msg)

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True)
    raise ValueError(f"Supabase initialization failed: {e}")

MAX_TOTAL_SESSIONS = 1000
SESSION_TIMEOUT_SEC = 3600 * 24

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class SessionData(BaseModel):
    history: List[ChatMessage] = Field(default_factory=list)
    last_accessed: float = Field(default_factory=time.time)

# --- Session / Logging helpers ---

def get_or_create_session_id(request: Request) -> str:
    session_id = request.session.get("chat_session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session["chat_session_id"] = session_id
    return session_id

def send_sse(data: Union[BaseModel, dict]) -> str:
    if isinstance(data, BaseModel):
        json_str = data.model_dump_json(by_alias=True)
    else:
        json_str = json.dumps(data, ensure_ascii=False)
    return f"data: {json_str}\n\n"

def log_context(
    session_id: str, message: str, level: str = "info", exc_info: bool = False
):
    safe_message = message.replace("\n", "\\n").replace("\r", "\\r")
    msg = f"[Session: {session_id}] {safe_message}"
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(msg, exc_info=exc_info)

class ChatHistoryManager:
    def __init__(self, max_length: int = 20):
        self._store: dict[str, SessionData] = {}
        self.max_length = max_length

    def _cleanup(self):
        current_time = time.time()
        expired = [
            sid
            for sid, data in self._store.items()
            if current_time - data.last_accessed > SESSION_TIMEOUT_SEC
        ]
        for sid in expired:
            del self._store[sid]
        if len(self._store) > MAX_TOTAL_SESSIONS:
            sorted_sessions = sorted(
                self._store.items(), key=lambda x: x[1].last_accessed
            )
            excess = len(self._store) - MAX_TOTAL_SESSIONS
            for i in range(excess):
                del self._store[sorted_sessions[i][0]]

    def add(self, session_id: str, role: str, content: str):
        if len(self._store) >= MAX_TOTAL_SESSIONS:
            self._cleanup()
        if session_id not in self._store:
            self._store[session_id] = SessionData()
        session = self._store[session_id]
        session.last_accessed = time.time()
        session.history.append(ChatMessage(role=role, content=content))
        if len(session.history) > self.max_length:
            session.history = session.history[-self.max_length :]

    def get_history(self, session_id: str) -> List[dict]:
        if session_id in self._store:
            session = self._store[session_id]
            session.last_accessed = time.time()
            return [msg.model_dump() for msg in session.history]
        return []


def format_urls_as_links(text: str) -> str:
    if not text:
        return ""
    url_pattern = (
        r"(?<!\()(https?://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|])"
    )
    def replace_link(match):
        url = match.group(0)
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return url
        except Exception:
            return url
        return f"[{url}]({url})"
    return re.sub(url_pattern, replace_link, text)


# ─────────────────────────────────────────────────────────
# ストレージ URL 生成ロジック
# ─────────────────────────────────────────────────────────

# Supabase Storage で有効なキー: ASCII英数字・ハイフン・アンダースコア・ピリオド・スラッシュ
# 日本語等のマルチバイト文字を含むキーは InvalidKey になるため事前にはじく
_ALLOWED_KEY_RE = re.compile(r'^[A-Za-z0-9_./()\-]+$')

# ★ ハッシュ値っぽいパス（16進数32文字 + 拡張子）かどうかを判定
_HASH_LIKE_RE = re.compile(r'^[0-9a-f]{8,}[^/]*\.[a-z]{2,5}$', re.IGNORECASE)


def _is_valid_storage_key(path: str) -> bool:
    """
    Supabase Storage キーとして安全に使えるパスかを判定する。
    - マルチバイト文字（日本語など）を含む場合は False
    - http/https で始まる URL は False（既に URL が分かっている場合）
    """
    if not path:
        return False
    if path.startswith(("http://", "https://")):
        return False
    # ASCII 範囲外の文字が含まれていたら NG
    try:
        path.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def generate_storage_url(file_path: str) -> Optional[str]:
    """
    Supabase Storage の署名付きURL（有効期限1時間）を生成する。

    ★ 修正ポイント:
    - 日本語等のマルチバイト文字を含むパスは試みずに None を返す（InvalidKey 防止）
    - 存在しないキーや 400/404 エラーは DEBUG ログのみ（スタックトレース抑制）
    """
    if not file_path or not supabase:
        return None

    # マルチバイト文字チェック（日本語ファイル名は Storage キーとして使えない）
    if not _is_valid_storage_key(file_path):
        logger.debug(
            f"Storage URL skipped (non-ASCII path): {file_path!r}"
        )
        return None

    # 拡張子チェック（ホワイトリスト）
    allowed_extensions = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf")
    if not file_path.lower().endswith(allowed_extensions):
        logger.debug(f"Storage URL skipped (unsupported ext): {file_path!r}")
        return None

    try:
        res = supabase.storage.from_(STORAGE_BUCKET_NAME).create_signed_url(
            file_path, 3600
        )
        if isinstance(res, dict) and "signedURL" in res:
            return res["signedURL"]
        elif isinstance(res, str) and res.startswith("http"):
            return res
        signed_url = getattr(res, "signed_url", None) or getattr(res, "signedURL", None)
        if isinstance(signed_url, str):
            return signed_url
        return None

    except Exception as e:
        # ★ スタックトレースなしの DEBUG ログに変更（ノイズ削減）
        err_msg = str(e)
        if "not_found" in err_msg or "404" in err_msg:
            logger.debug(f"Storage object not found: {file_path!r}")
        elif "InvalidKey" in err_msg or "400" in err_msg:
            logger.debug(f"Storage invalid key: {file_path!r}")
        else:
            logger.warning(f"Storage URL error ({file_path!r}): {err_msg}")
        return None


def format_references(documents: List[Any]) -> str:
    """
    RAG検索結果から参照元リストを生成。

    ★ 修正ポイント:
    - metadata.image_path（ハッシュ値）があればそれで URL を試みる
    - image_path がない場合は source（日本語表示名）を URL 生成に使わない
      → リンクなしでファイル名のみ表示
    """
    if not documents:
        return ""

    formatted_lines = ["\n\n## 参照元"]
    seen_sources = set()
    index = 1

    for doc in documents:
        metadata: dict = {}
        if isinstance(doc, dict):
            metadata = doc.get("metadata", {}) or {}
        elif hasattr(doc, "metadata"):
            m = getattr(doc, "metadata", {})
            metadata = m if isinstance(m, dict) else {}

        # 表示用ファイル名（source = 日本語表示名）
        source_name = str(metadata.get("source", "資料名不明"))
        display_name = os.path.basename(source_name)

        # ★ URL 生成には image_path（ハッシュ値）を優先使用
        #    source は表示専用・Storage キーとしては使わない
        image_path = metadata.get("image_path")
        url = metadata.get("url")  # キャッシュ済み URL があればそれを使う

        # image_path がある場合のみ署名付きURL を試みる
        if not url and image_path:
            url = generate_storage_url(image_path)

        # URL バリデーション
        if url:
            try:
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https"):
                    url = None
            except Exception:
                url = None

        unique_key = url if url else display_name
        if unique_key in seen_sources:
            continue
        seen_sources.add(unique_key)

        # Markdown エスケープ
        safe_display_name = display_name.replace("[", "\\[").replace("]", "\\]")

        if url:
            line = f"* [{index}] [{safe_display_name}]({url}) ⏳1時間有効"
        else:
            # URL が取得できない場合はリンクなし（日本語ファイル名でも表示は維持）
            line = f"* [{index}] {safe_display_name}"

        formatted_lines.append(line)
        index += 1

    if len(formatted_lines) > 1:
        return "\n".join(formatted_lines)
    return ""