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

# ロガーの設定
logger = logging.getLogger(__name__)

# --- 設定: 環境変数から取得 ---
STORAGE_BUCKET_NAME = os.getenv("SUPABASE_STORAGE_BUCKET", "slides")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# --- 【指針準拠修正】Fail Fast: 必須変数がなければ起動停止 ---
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    # AI_CONTEXT.md "Fail Fast" 準拠: 必須環境変数の欠落はエラーにする
    error_msg = "Critical Error: SUPABASE_URL or SUPABASE_SERVICE_KEY is missing."
    logger.critical(error_msg)
    raise ValueError(error_msg)

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
except Exception as e:
    # 初期化失敗時も即停止
    logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True)
    raise ValueError(f"Supabase initialization failed: {e}")

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

# --- 【機能修正】URL生成ロジック ---

def generate_storage_url(source_name: str) -> Optional[str]:
    """
    Supabase Storageの署名付きURL（有効期限1時間）を生成する。
    Args:
        source_name: DBのmetadata['source'] (パスを含むファイル名)
    """
    if not source_name or not supabase:
        return None

    # 【修正点】フォルダ構造を維持するため、basename処理を削除
    # DBのsourceに 'folder/file.jpg' と入っている場合、そのままパスとして使用する
    file_path = source_name

    # 拡張子チェック（ホワイトリスト方式）
    allowed_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.webp', '.pdf')
    if not file_path.lower().endswith(allowed_extensions):
        # 画像・PDF以外は生成しない（必要に応じて緩和してください）
        return None

    try:
        # ▼▼▼ 追加・修正箇所 ▼▼▼
        
        # 日本語ファイル名が含まれるとAPIが400エラーになるためURLエンコードする
        encoded_path = quote(file_path, safe='/')

        # 署名付きURL生成
        res = supabase.storage.from_(STORAGE_BUCKET_NAME).create_signed_url(
            encoded_path,  # file_path ではなく encoded_path を渡す
            3600
        )
        
        # ▲▲▲ 追加・修正箇所終わり ▲▲▲

        # レスポンスハンドリング（Strict Typing: バージョン差分吸収）
        if isinstance(res, dict) and 'signedURL' in res:
            return res['signedURL']
        elif isinstance(res, str) and res.startswith('http'):
            return res
        # Pydanticモデル等が返ってきた場合のハンドリング
        signed_url = getattr(res, 'signed_url', None) or getattr(res, 'signedURL', None)
        if isinstance(signed_url, str):
            return signed_url
            
        return None

    except Exception as e:
        # AI_CONTEXT.md "Robustness": スタックトレースを含めてログ出力
        logger.warning(f"Failed to generate signed URL for {source_name}: {e}", exc_info=True)
        return None


def format_references(documents: List[Any]) -> str:
    """
    RAG検索結果から参照元リストを生成。
    """
    if not documents:
        return ""

    formatted_lines = ["\n\n## 参照元 (クリックで資料を表示・1時間有効)"]
    seen_sources = set()
    index = 1

    for doc in documents:
        # 【指針準拠】型判定を強化して安全にアクセス
        metadata = {}
        if isinstance(doc, dict):
            metadata = doc.get("metadata", {})
        elif hasattr(doc, "metadata"):
            # LangChain Document object
            m = getattr(doc, "metadata", {})
            metadata = m if isinstance(m, dict) else {}
        
        # ソース取得
        source_name = str(metadata.get("source", "資料名不明"))
        
        # 表示名はファイル名のみにする（パスを除去）
        display_name = os.path.basename(source_name)
        
        url = metadata.get("url")
        
        # URL自動生成の試行
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

        # Markdownエスケープ
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