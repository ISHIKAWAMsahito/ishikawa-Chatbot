import json
import uuid
import logging
from typing import Dict, Any, List
from fastapi import Request
import re

def get_or_create_session_id(request: Request) -> str:
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    return session_id

def send_sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

def log_context(session_id: str, message: str, level: str = "info"):
    msg = f"[Session: {session_id}] {message}"
    getattr(logging, level, logging.info)(msg)

class ChatHistoryManager:
    def __init__(self, max_length: int = 20):
        self._histories: Dict[str, List[Dict[str, str]]] = {}
        self.max_length = max_length

    def add(self, session_id: str, role: str, content: str):
        if session_id not in self._histories:
            self._histories[session_id] = []
        self._histories[session_id].append({"role": role, "content": content})
        if len(self._histories[session_id]) > self.max_length:
            self._histories[session_id] = self._histories[session_id][-self.max_length:]

def format_urls_as_links(text: str) -> str:
    """
    テキスト内のURLを検出し、Markdownのリンク形式 [URL](URL) に変換する。
    """
    if not text:
        return ""
    # URL検出用の正規表現
    url_pattern = r'(?<!\()https?://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]'
    def replace_link(match):
        url = match.group(0)
        return f"[{url}]({url})"

    # 既にMarkdownリンクの形式になっているURL (例: ](http...)) を避けるための簡易的な対策として
    # 否定後読み (?<!\() を入れていますが、完全ではありません。
    # 必要に応じて調整してください。
    return re.sub(url_pattern, replace_link, text)