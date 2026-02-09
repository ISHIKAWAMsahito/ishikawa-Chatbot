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

# ▼ 追加: Supabaseクライアントのインポート
from supabase import create_client, Client

# ロガーの設定
logger = logging.getLogger(__name__)

# --- 設定: 環境変数から取得 ---
STORAGE_BUCKET_NAME = os.getenv("SUPABASE_STORAGE_BUCKET", "slides") 
SUPABASE_URL = os.getenv("SUPABASE_URL")
# ▼ 重要: 署名付きURLの発行にはService Key（または適切な権限を持つAnon Key）が必要です
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# ▼ クライアントの初期化（キーがない場合のガード付き）
supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")

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

# --- Functions (Session / Logging) ---
# ... (既存の get_or_create_session_id, send_sse, log_context, ChatHistoryManager は変更なし) ...

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

# --- ✨ 変更: 署名付きURL生成ロジック ---

def generate_storage_url(source_name: str) -> Optional[str]:
    """
    Supabase Storageの署名付きURL（有効期限1時間）を生成する。
    Args:
        source_name: DBのmetadata['source'] (例: '20251226.jpg')
    Returns:
        有効な署名付きURL または None
    """
    if not source_name or not supabase:
        return None

    # パストラバーサル対策
    safe_filename = os.path.basename(source_name)
    
    # 拡張子チェック
    if not any(safe_filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
        return None

    # ▼ ここでパスを調整してください
    # 画像が 'images' フォルダ内にある場合は f"images/{safe_filename}" とします
    # source_name 自体がパスを含んでいる場合はそのまま使うことも検討してください
    file_path = safe_filename 
    # file_path = f"images/{safe_filename}"  # フォルダが必要な場合

    try:
        # 🔑 3600秒（1時間）有効なURLを作成
        res = supabase.storage.from_(STORAGE_BUCKET_NAME).create_signed_url(
            file_path, 
            3600
        )
        # レスポンス形式: {'signedURL': 'https://...', ...} (v2系)
        # バージョンによって形式が異なる場合があるため調整
        if isinstance(res, dict) and 'signedURL' in res:
            return res['signedURL']
        elif isinstance(res, str): # 古いバージョンやエラー文字列
             return res
        else:
             # オブジェクトで返ってくる場合（最新の supabase-py）
             return getattr(res, 'signed_url', None) or res.get('signedURL')

    except Exception as e:
        logger.warning(f"Failed to generate signed URL for {source_name}: {e}")
        return None


def format_references(documents: List[object]) -> str:
    """
    RAG検索結果から参照元リストを生成。
    URLがない場合は署名付きURLの自動生成を試みる。
    """
    if not documents:
        return ""

    formatted_lines = ["\n\n## 参照元 (クリックで資料を表示・1時間有効)"]
    seen_sources = set()
    index = 1

    for doc in documents:
        if isinstance(doc, dict):
            metadata = doc.get("metadata", {})
        else:
            metadata = getattr(doc, "metadata", {})
            if not isinstance(metadata, dict):
                 metadata = metadata if metadata else {}

        source_name = str(metadata.get("source", "資料名不明"))
        display_name = os.path.basename(source_name)
        
        url = metadata.get("url")
        
        # URLがない、または空の場合は署名付きURLを生成
        if not url and source_name != "資料名不明":
            url = generate_storage_url(source_name)

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