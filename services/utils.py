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

# 定数設定
MAX_TOTAL_SESSIONS = 1000  # メモリ保護（DoS対策）
SESSION_TIMEOUT_SEC = 3600 * 24  # 24時間で自動削除

# --- Pydantic Models (Dict禁止ルールへの準拠) ---
class ChatMessage(BaseModel):
    role: str
    content: str

class SessionData(BaseModel):
    history: List[ChatMessage] = Field(default_factory=list)
    last_accessed: float = Field(default_factory=time.time)

# --- Functions ---

def get_or_create_session_id(request: Request) -> str:
    """
    セッションIDを取得、なければ生成して管理
    Context: 認証関数やミドルウェアと連携して安全性を担保すること
    """
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    return session_id

def send_sse(data: Union[BaseModel, dict]) -> str:
    """
    Server-Sent Events形式のデータを作成する
    Context: Pydanticモデルを優先的に受け取り、JSONシリアライズする
    """
    if isinstance(data, BaseModel):
        # キャメルケースへの自動変換設定があれば by_alias=True が効く
        json_str = data.model_dump_json(by_alias=True)
    else:
        # レガシー対応（dictが渡された場合）
        json_str = json.dumps(data, ensure_ascii=False)
        
    return f"data: {json_str}\n\n"

def log_context(session_id: str, message: str, level: str = "info", exc_info: bool = False):
    """
    ログ・インジェクション対策済みのロガー
    Context: エラー時は exc_info=True を推奨
    """
    # 改行コードを無効化（ログ改ざん防止）
    safe_message = message.replace('\n', '\\n').replace('\r', '\\r')
    msg = f"[Session: {session_id}] {safe_message}"
    
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(msg, exc_info=exc_info)

class ChatHistoryManager:
    """
    メモリ枯渇(DoS)対策 & Pydantic型安全性を施した履歴マネージャー
    """
    def __init__(self, max_length: int = 20):
        # Dict[str, SessionData] として型定義
        self._store: dict[str, SessionData] = {}
        self.max_length = max_length

    def _cleanup(self):
        """ガベージコレクション: 古いセッションや上限超過分を削除"""
        current_time = time.time()
        
        # タイムアウト削除
        expired = [
            sid for sid, data in self._store.items() 
            if current_time - data.last_accessed > SESSION_TIMEOUT_SEC
        ]
        for sid in expired:
            del self._store[sid]

        # 上限数削除（LRU方式）
        if len(self._store) > MAX_TOTAL_SESSIONS:
            sorted_sessions = sorted(self._store.items(), key=lambda x: x[1].last_accessed)
            excess = len(self._store) - MAX_TOTAL_SESSIONS
            for i in range(excess):
                del self._store[sorted_sessions[i][0]]

    def add(self, session_id: str, role: str, content: str):
        """メッセージを追加する"""
        # メモリ保護チェック
        if len(self._store) >= MAX_TOTAL_SESSIONS:
            self._cleanup()

        if session_id not in self._store:
            self._store[session_id] = SessionData()
        
        session = self._store[session_id]
        session.last_accessed = time.time()
        
        # Pydanticモデルとして追加
        session.history.append(ChatMessage(role=role, content=content))

        # 履歴ローテーション
        if len(session.history) > self.max_length:
            session.history = session.history[-self.max_length:]

    def get_history(self, session_id: str) -> List[dict]:
        """
        APIレスポンス用に辞書のリストとして返す
        (LLM等の呼び出し元が dict 形式を期待している場合に対応)
        """
        if session_id in self._store:
            session = self._store[session_id]
            session.last_accessed = time.time()
            # Pydanticモデルを辞書化して返す
            return [msg.model_dump() for msg in session.history]
        return []

def format_urls_as_links(text: str) -> str:
    """
    テキスト内のURLを検出し、安全なMarkdownリンクに変換する。
    XSS対策: http/https スキームのみ許可
    """
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

def format_references(documents: List[object]) -> str:
    """
    RAG検索結果から参照元リストを生成。
    Path Traversal / XSS 対策済み
    Args:
        documents: LangChain Document オブジェクト または Pydantic モデルのリスト
    """
    if not documents:
        return ""

    formatted_lines = ["\n\n## 参照元 (クリックで資料を表示)"]
    seen_sources = set()
    index = 1

    for doc in documents:
        # Pydantic, Object, Dict いずれにも対応できる安全な属性取得
        if isinstance(doc, dict):
            metadata = doc.get("metadata", {})
        else:
            metadata = getattr(doc, "metadata", {})
            if not isinstance(metadata, dict):
                 # LangChain Documentの場合 metadata属性自体がdict
                 metadata = metadata if metadata else {}

        source_name = str(metadata.get("source", "資料名不明"))
        display_name = os.path.basename(source_name)
        
        url = metadata.get("url")
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