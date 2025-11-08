import os
import logging
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
import google.generativeai as genai

from core.config import GEMINI_API_KEY, ACTIVE_COLLECTION_NAME
# ↓↓↓ [修正] モジュール本体をインポート
from core import database
# ↓↓↓ [修正] "as core_settings" という別名を付ける
from core import settings as core_settings
# ↑↑↑ [修正]
from models.schemas import Settings

router = APIRouter()

@router.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "ok",
        # ↓↓↓ [修正] 
        # db_client が存在する時点で "supabase" とわかるので、ハードコードする
        "database": "supabase" if database.db_client else "uninitialized"
        # ↑↑↑ [修正]
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
        # ↓↓↓ [修正] "database." を追加
        "count": database.db_client.count_chunks_in_collection(ACTIVE_COLLECTION_NAME) if database.db_client else 0
        # ↑↑↑ [修正]
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
# ↓↓↓ [修正] 引数名を "settings_payload" に変更
async def update_settings(settings_payload: Settings):
    """設定を更新"""
    # ↓↓↓ [修正] 別名を付けた "core_settings" を参照する
    if not core_settings.settings_manager:
        raise HTTPException(503, "設定マネージャーが初期化されていません")
    try:
        # ↓↓↓ [修正] "core_settings" を参照し、引数のPydanticモデルを渡す
        await core_settings.settings_manager.update_settings(settings_payload.dict(exclude_none=True))
        return {"message": "設定を更新しました"}
    except Exception as e:
        logging.error(f"設定更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/settings")
async def websocket_endpoint(websocket: WebSocket):
    """設定変更通知用WebSocket"""
    # ↓↓↓ [修正] 別名を付けた "core_settings" を参照する
    if not core_settings.settings_manager:
        await websocket.close(code=1011, reason="Settings manager not initialized")
        return
    await core_settings.settings_manager.add_websocket(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        # ↓↓↓ [修正] 別名を付けた "core_settings" を参照する
        core_settings.settings_manager.remove_websocket(websocket)