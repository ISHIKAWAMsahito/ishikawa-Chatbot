import logging
import jwt
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import google.generativeai as genai

# Config & Dependencies
from core.config import (
    GEMINI_API_KEY, 
    ACTIVE_COLLECTION_NAME, 
    SUPABASE_URL, 
    SUPABASE_ANON_KEY,
    SECRET_KEY
)
from core.dependencies import require_auth
from core import database
from core import settings as core_settings
from models.schemas import Settings, CreateCollectionRequest

router = APIRouter()
logger = logging.getLogger(__name__)
GENERIC_ERROR_MSG = "処理に失敗しました。"

# ---------------------------------------------------------
# Pydantic Response Models (Strict Typing)
# ---------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    database: str

class GeminiStatusResponse(BaseModel):
    connected: bool
    models: Optional[List[str]] = None
    detail: Optional[str] = None

class ConfigResponse(BaseModel):
    supabase_url: str
    supabase_anon_key: str

class CollectionItem(BaseModel):
    name: str
    count: int

class WSTokenResponse(BaseModel):
    token: str
    expires_in_seconds: int

class MessageResponse(BaseModel):
    message: str

# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """ヘルスチェック"""
    return HealthResponse(
        status="ok",
        database="Supabase" if database.db_client else "uninitialized"
    )

@router.api_route("/healthz", methods=["GET", "HEAD"])
def health_check_k8s():
    """Kubernetes用ヘルスチェック"""
    return {"status": "ok"}

@router.get("/gemini/status", response_model=GeminiStatusResponse)
async def gemini_status(user: Any = Depends(require_auth)):
    """Gemini APIの接続状態を確認（管理者用）"""
    if not GEMINI_API_KEY:
        return GeminiStatusResponse(connected=False, detail="API key not configured")
    
    try:
        # 生成可能なモデルのみをリストアップ
        models = [
            m.name for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        ]
        return GeminiStatusResponse(connected=True, models=models)
    except Exception as e:
        logging.error(f"Gemini status error: {e}", exc_info=True)
        return GeminiStatusResponse(connected=False, detail="接続確認に失敗しました。")

@router.get("/config", response_model=ConfigResponse)
def get_config(user: Any = Depends(require_auth)):
    """フロントエンドが必要とする設定を返す（管理者用）"""
    # 必須設定がない場合は空文字を返すなどのハンドリングも可能だが、
    # Fail Fastにより起動時に保証されている前提とする
    return ConfigResponse(
        supabase_url=SUPABASE_URL or "",
        supabase_anon_key=SUPABASE_ANON_KEY or ""
    )

@router.get("/collections", response_model=List[CollectionItem])
async def get_collections(user: Any = Depends(require_auth)):
    """コレクション一覧を取得"""
    count = 0
    if database.db_client:
        # DB接続がある場合のみカウントを取得
        try:
            count = database.db_client.count_chunks_in_collection(ACTIVE_COLLECTION_NAME)
        except Exception as e:
            logging.error(f"Count collection error: {e}")
            count = 0
            
    return [
        CollectionItem(name=ACTIVE_COLLECTION_NAME, count=count)
    ]

@router.post("/collections", response_model=MessageResponse)
async def create_collection(request: CreateCollectionRequest, user: Any = Depends(require_auth)):
    """コレクションを作成(既存のみ)"""
    return MessageResponse(message=f"コレクション「{ACTIVE_COLLECTION_NAME}」は既に存在しています")

@router.get("/ws-token", response_model=WSTokenResponse)
def get_ws_token(user: Any = Depends(require_auth)):
    """
    WebSocket /ws/settings 接続用の短期トークンを発行（管理者認証必須）。
    検証側 (core/ws_auth.py) と同じ SECRET_KEY で署名する。
    """
    try:
        now = datetime.now(timezone.utc)
        expires_in = 60 # 秒
        
        payload = {
            "sub": "admin-ws",
            "iat": now,
            "exp": now + timedelta(seconds=expires_in)
        }
        
        # 403回避の要: ConfigのSECRET_KEYを使用
        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        
        return WSTokenResponse(token=token, expires_in_seconds=expires_in)
    except Exception as e:
        logging.error(f"Token generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="トークン生成に失敗しました")

@router.delete("/collections/{collection_name}", response_model=MessageResponse)
async def delete_collection(collection_name: str, user: Any = Depends(require_auth)):
    """コレクションを削除"""
    if collection_name == ACTIVE_COLLECTION_NAME:
        raise HTTPException(status_code=400, detail="このコレクションは削除できません")
    return MessageResponse(message="コレクションが見つかりません")

@router.post("/settings", response_model=MessageResponse)
async def update_settings(settings_payload: Settings, user: Any = Depends(require_auth)):
    """設定を更新"""
    if not core_settings.settings_manager:
        raise HTTPException(status_code=503, detail="設定マネージャーが初期化されていません")
    try:
        await core_settings.settings_manager.update_settings(settings_payload.model_dump(exclude_none=True))
        return MessageResponse(message="設定を更新しました")
    except Exception as e:
        logging.error(f"設定更新エラー: {e}", exc_info=True)
        # 指針: 詳細エラーをクライアントに返さない
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_MSG)