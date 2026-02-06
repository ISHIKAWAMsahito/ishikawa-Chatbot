import json
import uuid
import logging
import os
import re
from typing import Dict, Any, List
from fastapi import Request

def get_or_create_session_id(request: Request) -> str:
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    return session_id

def send_sse(data: Dict[str, Any]) -> str:
    """Server-Sent Events形式のデータを作成する"""
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

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self._histories.get(session_id, [])

def format_urls_as_links(text: str) -> str:
    """
    テキスト内のURLを検出し、Markdownのリンク形式 [URL](URL) に変換する。
    （AIの回答本文内のURL用）
    """
    if not text:
        return ""
    # URL検出用の正規表現
    url_pattern = r'(?<!\()https?://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]'
    
    def replace_link(match):
        url = match.group(0)
        return f"[{url}]({url})"

    return re.sub(url_pattern, replace_link, text)

def format_references(documents: List[Dict[str, Any]]) -> str:
    """
    RAG検索結果のドキュメントリストから、Markdown形式の参照元リストを生成する。
    metadataに 'url' が含まれる場合はハイパーリンク化する。
    
    Args:
        documents: 検索サービスから返却されたドキュメントのリスト
        
    Returns:
        str: Markdown形式の参照元リスト文字列
    """
    if not documents:
        return ""

    formatted_lines = ["\n\n## 参照元 (クリックで資料を表示)"]
    
    # 重複排除用（同じURLやファイルが複数チャンク引っかかる場合があるため）
    seen_sources = set()
    index = 1

    for doc in documents:
        # docがオブジェクトか辞書かで取得方法を分岐
        if isinstance(doc, dict):
            metadata = doc.get("metadata", {})
        else:
            metadata = getattr(doc, "metadata", {})

        source_name = metadata.get("source", "資料名不明")
        url = metadata.get("url")

        # ファイルパスが含まれる場合はファイル名のみ抽出
        display_name = os.path.basename(source_name)

        # ユニークキーを作成（同じファイル/URLを何度も表示しないようにする）
        unique_key = url if url else display_name
        
        if unique_key in seen_sources:
            continue
        
        seen_sources.add(unique_key)

        if url:
            # URLがある場合: [番号] [タイトル](URL) の形式
            line = f"* [{index}] [{display_name}]({url})"
        else:
            # URLがない場合: [番号] タイトル の形式
            line = f"* [{index}] {display_name}"

        formatted_lines.append(line)
        index += 1

    # 参照元が1つ以上ある場合のみ文字列を返す
    if len(formatted_lines) > 1:
        return "\n".join(formatted_lines)
    
    return ""