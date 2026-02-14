import json
import uuid
import logging
import os
import re
import time
from typing import List, Union, Optional, Any, Dict # Any追加
from urllib.parse import urlparse
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


# ... (定数定義、ChatMessage, SessionData, get_or_create_session_id, send_sse, log_context, ChatHistoryManager は変更なし) ...
# ... (format_urls_as_links も変更なし) ...


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
        # 署名付きURL生成
        res = supabase.storage.from_(STORAGE_BUCKET_NAME).create_signed_url(
            file_path,
            3600
        )

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