import logging
from fastapi import APIRouter, HTTPException, Depends

import google.generativeai as genai

from core.config import GEMINI_API_KEY, ACTIVE_COLLECTION_NAME, SUPABASE_URL, SUPABASE_ANON_KEY
from core.dependencies import require_auth
from core import database
from core import settings as core_settings
from models.schemas import Settings, CreateCollectionRequest
from core.ws_auth import create_ws_token

router = APIRouter()
GENERIC_ERROR_MSG = "処理に失敗しました。"

# ヘルスチェックや設定管理を行います。

# ヘルスチェック: DB接続やGemini APIのステータスを確認します。Kubernetes用の /healthz もあります。

# 設定変更: 使用するAIモデル（gemini-2.5-flashなど）やRAGのパラメータを動的に変更するAPIを提供しています。


@router.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "ok",
        "database": "Supabase" if database.db_client else "uninitialized"
    }

@router.api_route("/healthz", methods=["GET", "HEAD"])
def health_check_k8s():
    """Kubernetes用ヘルスチェック"""
    return {"status": "ok"}

@router.get("/gemini/status")
async def gemini_status(user: dict = Depends(require_auth)):
    """Gemini APIの接続状態を確認（管理者用）"""
    if not GEMINI_API_KEY:
        return {"connected": False, "detail": "API key not configured"}
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return {"connected": True, "models": models}
    except Exception as e:
        logging.error(f"Gemini status error: {e}", exc_info=True)
        return {"connected": False, "detail": "接続確認に失敗しました。"}

@router.get("/config")
def get_config(user: dict = Depends(require_auth)):
    """フロントエンドが必要とする設定を返す（管理者用・core.config の定数を使用）"""
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY,
    }

@router.get("/collections")
async def get_collections(user: dict = Depends(require_auth)):
    """コレクション一覧を取得"""
    return [{
        "name": ACTIVE_COLLECTION_NAME,
        "count": database.db_client.count_chunks_in_collection(ACTIVE_COLLECTION_NAME) if database.db_client else 0
    }]

@router.post("/collections")
async def create_collection(request: CreateCollectionRequest, user: dict = Depends(require_auth)):
    """コレクションを作成(既存のみ)。リクエストは Pydantic 必須（dict 禁止）。"""
    return {"message": f"コレクション「{ACTIVE_COLLECTION_NAME}」は既に存在しています"}


@router.get("/ws-token")
def get_ws_token(user: dict = Depends(require_auth)):
    """WebSocket /ws/settings 接続用の短期トークンを発行（管理者認証必須）。60秒有効。"""
    token = create_ws_token()
    return {"token": token, "expires_in_seconds": 60}

@router.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str, user: dict = Depends(require_auth)):
    """コレクションを削除"""
    if collection_name == ACTIVE_COLLECTION_NAME:
        raise HTTPException(status_code=400, detail="このコレクションは削除できません")
    return {"message": "コレクションが見つかりません"}

@router.post("/settings")
async def update_settings(settings_payload: Settings, user: dict = Depends(require_auth)):
    """設定を更新"""
    if not core_settings.settings_manager:
        raise HTTPException(503, "設定マネージャーが初期化されていません")
    try:
        await core_settings.settings_manager.update_settings(settings_payload.dict(exclude_none=True))
        return {"message": "設定を更新しました"}
    except Exception as e:
        logging.error(f"設定更新エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=GENERIC_ERROR_MSG)

# WebSocketコードは main.py へ移動しました