import os
import logging
from fastapi import APIRouter, HTTPException
import google.generativeai as genai

from core.config import GEMINI_API_KEY, ACTIVE_COLLECTION_NAME
from core import database
from core import settings as core_settings
from models.schemas import Settings

router = APIRouter()

@router.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "ok",
        "database": "supabase" if database.db_client else "uninitialized"
    }

@router.api_route("/healthz", methods=["GET", "HEAD"])
def health_check_k8s():
    """Kubernetes用ヘルスチェック"""
    return {"status": "ok"}

@router.get("/gemini/status")
async def gemini_status():
    """Gemini APIの接続状態を確認"""
    if not GEMINI_API_KEY:
        return {"connected": False, "detail": "API key not configured"}
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return {"connected": True, "models": models}
    except Exception as e:
        return {"connected": False, "detail": str(e)}

@router.get("/config")
def get_config():
    """フロントエンドが必要とする公開可能な設定(環境変数)を返す"""
    return {
        "supabase_url": os.getenv("SUPABASE_URL"),
        "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY")
    }

@router.get("/collections")
async def get_collections():
    """コレクション一覧を取得"""
    return [{
        "name": ACTIVE_COLLECTION_NAME, 
        "count": database.db_client.count_chunks_in_collection(ACTIVE_COLLECTION_NAME) if database.db_client else 0
    }]

@router.post("/collections")
async def create_collection(request: dict):
    """コレクションを作成(既存のみ)"""
    return {"message": f"コレクション「{ACTIVE_COLLECTION_NAME}」は既に存在しています"}

@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """コレクションを削除"""
    if collection_name == ACTIVE_COLLECTION_NAME:
        raise HTTPException(status_code=400, detail="このコレクションは削除できません")
    return {"message": "コレクションが見つかりません"}

@router.post("/settings")
async def update_settings(settings_payload: Settings):
    """設定を更新"""
    if not core_settings.settings_manager:
        raise HTTPException(503, "設定マネージャーが初期化されていません")
    try:
        await core_settings.settings_manager.update_settings(settings_payload.dict(exclude_none=True))
        return {"message": "設定を更新しました"}
    except Exception as e:
        logging.error(f"設定更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocketコードは main.py へ移動しました