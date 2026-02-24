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
    Supabase Storage の署名付きURL（3600秒 = 1時間）を生成。
    フォールバック処理: .txt 等のファイルが指定された場合、同名の画像ファイルを探してURLを生成する。
    """
    if not file_path or not supabase:
        return None

    # 日本語等のマルチバイト文字が含まれるキーはエラーになるため除外
    if not _is_valid_storage_key(file_path):
        logger.debug(f"Storage URL skipped (invalid key): {file_path!r}")
        return None

    # 検索するファイルの候補リストを作成
    candidates = []
    base_name, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    # 許可されている画像（およびPDF）の拡張子リスト
    allowed_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".pdf"]
    
    if ext in allowed_extensions:
        # 元のファイルが画像/PDFなら、そのまま候補の筆頭にする
        candidates.append(file_path)
    elif ext in [".txt", ".md", ".csv"]:
        # .txt や .md などのテキストファイルの場合、画像拡張子に置換して探す（フォールバック）
        logger.debug(f"Fallback initiated for text file: {file_path!r}")
        for img_ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidates.append(f"{base_name}{img_ext}")
    else:
        # 対象外の拡張子の場合はスキップ
        return None

    # 候補のパス順に URL 生成を試みる
    for candidate_path in candidates:
        try:
            res = supabase.storage.from_(STORAGE_BUCKET_NAME).create_signed_url(
                candidate_path, 3600
            )
            
            # 戻り値のチェック
            signed_url = None
            if isinstance(res, dict) and "signedURL" in res:
                signed_url = res["signedURL"]
            elif isinstance(res, str) and res.startswith("http"):
                signed_url = res
            else:
                s_url = getattr(res, "signed_url", None) or getattr(res, "signedURL", None)
                if s_url:
                    signed_url = str(s_url)
            
            if signed_url:
                # 成功したらその URL を返す
                if candidate_path != file_path:
                    logger.info(f"Fallback successful: {file_path} -> {candidate_path}")
                return signed_url
                
        except Exception as e:
            # 存在しない場合などはエラーになるため、無視して次の候補を探す
            err_msg = str(e)
            if "not_found" in err_msg or "404" in err_msg:
                logger.debug(f"Storage object not found during fallback: {candidate_path!r}")
            else:
                logger.debug(f"Fallback generation failed for {candidate_path!r}: {e}")
            continue

    # 全ての候補で生成失敗した場合
    logger.warning(f"Failed to generate signed URL for any candidates of {file_path!r}")
    return None
def format_references(documents: List[Any]) -> str:
    """
    参照元リストの生成（文言の削除と画像URL生成の確実化）。
    """
    if not documents:
        return ""

    formatted_lines = ["\n\n## 参照元"]
    seen_sources = set()
    index = 1

    for doc in documents:
        metadata = doc.get("metadata", {}) if isinstance(doc, dict) else getattr(doc, "metadata", {})
        
        source_name = str(metadata.get("source", "資料名不明"))
        display_name = os.path.basename(source_name)
        image_path = metadata.get("image_path")
        
        # 修正点：画像(jpg/png等)の場合は、既存の url よりも新規署名付きURLの生成を優先する
        url = None
        if image_path:
            url = generate_storage_url(image_path)
        
        # 画像パスがない、または生成に失敗した場合のみ、既存の url フィールドを参照
        if not url:
            url = metadata.get("url")

        # 重複チェック
        unique_key = url if url else display_name
        if unique_key in seen_sources:
            continue
        seen_sources.add(unique_key)

        safe_display_name = display_name.replace("[", "\\[").replace("]", "\\]")

        # 修正点：「⏳1時間有効」の文言を削除
        if url:
            line = f"* [{index}] [{safe_display_name}]({url})"
        else:
            line = f"* [{index}] {safe_display_name}"

        formatted_lines.append(line)
        index += 1

    return "\n".join(formatted_lines) if len(formatted_lines) > 1 else ""
    return ""