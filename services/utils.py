import json
import uuid
import logging
import os
import re
import time
from typing import List, Union, Optional
from urllib.parse import urlparse
from fastapi import Request
from pydantic import BaseModel, Field

# ロガーの設定
logger = logging.getLogger(__name__)

# --- 設定: 環境変数から取得 ---
# ⚠️重要: ここに使用するバケット名を設定してください（例: "slides", "images", "documents"）
# システム構成書に基づき、画像が格納されているバケット名を指定します
STORAGE_BUCKET_NAME = os.getenv("SUPABASE_STORAGE_BUCKET", "slides") 
SUPABASE_URL = os.getenv("SUPABASE_URL")

# 定数設定
MAX_TOTAL_SESSIONS = 1000
SESSION_TIMEOUT_SEC = 3600 * 24

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class SessionData(BaseModel):
    history: List[ChatMessage] = Field(default_factory=list)
    last_accessed: float = Field(default_factory=time.time)

# --- Functions ---

def get_or_create_session_id(request: Request) -> str:
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    return session_id

def send_sse(data: Union[BaseModel, dict]) -> str:
    if isinstance(data, BaseModel):
        json_str = data.model_dump_json(by_alias=True)
    else:
        json_str = json.dumps(data, ensure_ascii=False)
    return f"data: {json_str}\n\n"

def log_context(session_id: str, message: str, level: str = "info", exc_info: bool = False):
    safe_message = message.replace('\n', '\\n').replace('\r', '\\r')
    msg = f"[Session: {session_id}] {safe_message}"
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(msg, exc_info=exc_info)

class ChatHistoryManager:
    def __init__(self, max_length: int = 20):
        self._store: dict[str, SessionData] = {}
        self.max_length = max_length

    def _cleanup(self):
        current_time = time.time()
        expired = [sid for sid, data in self._store.items() if current_time - data.last_accessed > SESSION_TIMEOUT_SEC]
        for sid in expired:
            del self._store[sid]
        
        if len(self._store) > MAX_TOTAL_SESSIONS:
            sorted_sessions = sorted(self._store.items(), key=lambda x: x[1].last_accessed)
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
            session.history = session.history[-self.max_length:]

    def get_history(self, session_id: str) -> List[dict]:
        if session_id in self._store:
            session = self._store[session_id]
            session.last_accessed = time.time()
            return [msg.model_dump() for msg in session.history]
        return []

def format_urls_as_links(text: str) -> str:
    if not text:
        return ""
    url_pattern = r'(?<!\()(https?://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|])'
    def replace_link(match):
        url = match.group(0)
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                return url
        except Exception:
            return url
        return f"[{url}]({url})"
    return re.sub(url_pattern, replace_link, text)

# --- ✨ 新規追加: 画像URL生成ロジック ---
def generate_storage_url(source_name: str) -> Optional[str]:
    """
    ファイル名からSupabase Storageの公開URLを生成する。
    Args:
        source_name: DBのmetadata['source'] (例: '20251226.jpg')
    Returns:
        有効なURL文字列 または None
    """
    if not source_name or not SUPABASE_URL:
        return None

    # セキュリティ: パストラバーサル対策 (../ を無効化し、ファイル名のみ抽出)
    safe_filename = os.path.basename(source_name)
    
    # 拡張子チェック (画像かどうか)
    if not any(safe_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
        return None

    # URL組み立て: {SUPABASE_URL}/storage/v1/object/public/{BUCKET}/{FILENAME}
    # ※ バケットが Public 設定であることを前提としています
    # ※ フォルダ構造がある場合はここで調整 (例: f"images/{safe_filename}")
    return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET_NAME}/{safe_filename}"


def format_references(documents: List[object]) -> str:
    """
    RAG検索結果から参照元リストを生成。
    URLがない場合は source から自動生成を試みる。
    """
    if not documents:
        return ""

    formatted_lines = ["\n\n## 参照元 (クリックで資料を表示)"]
    seen_sources = set()
    index = 1

    for doc in documents:
        # メタデータの取得
        if isinstance(doc, dict):
            metadata = doc.get("metadata", {})
        else:
            metadata = getattr(doc, "metadata", {})
            if not isinstance(metadata, dict):
                 metadata = metadata if metadata else {}

        source_name = str(metadata.get("source", "資料名不明"))
        display_name = os.path.basename(source_name)
        
        # --- 🛠 修正: URL取得ロジックの強化 ---
        url = metadata.get("url")
        
        # URLが空の場合、ファイル名から自動生成を試みる
        if not url and source_name != "資料名不明":
            url = generate_storage_url(source_name)
            if url:
                logger.info(f"Generated URL for {source_name}: {url}") # デバッグ用ログ

        # URLバリデーション
        if url:
            try:
                parsed = urlparse(url)
                if parsed.scheme not in ('http', 'https'):
                    url = None
            except Exception:
                url = None

        unique_key = url if url else display_name
        
        if unique_key in seen_sources:
            continue
        
        seen_sources.add(unique_key)

        safe_display_name = display_name.replace("[", "\\[").replace("]", "\\]")

        if url:
            line = f"* [{index}] [{safe_display_name}]({url})"
        else:
            line = f"* [{index}] {safe_display_name}"

        formatted_lines.append(line)
        index += 1

    if len(formatted_lines) > 1:
        return "\n".join(formatted_lines)
    
    return ""