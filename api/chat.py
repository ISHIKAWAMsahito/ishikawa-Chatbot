# api/chat.py

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from core.dependencies import require_auth_client
from core import settings as core_settings
from core.config import ACTIVE_COLLECTION_NAME
from models.schemas import ChatQuery, ClientChatQuery
from services.chat_logic import enhanced_chat_logic, get_or_create_session_id, history_manager
import logging

# ★重要: ルーターを2つ作成する
admin_router = APIRouter()   # 管理者用
client_router = APIRouter()  # 学生用

# ============================
# 管理者用エンドポイント (admin_router を使用)
# ============================
@admin_router.post("/")
async def chat_endpoint(request: Request, query: ChatQuery):
    """管理者用チャットエンドポイント"""
    return StreamingResponse(enhanced_chat_logic(request, query), media_type="text/event-stream")

# 他の管理者用APIがあれば admin_router に追加
# @admin_router.get("/history")... など

# ============================
# 学生用エンドポイント (client_router を使用)
# ============================
@client_router.post("/")
async def chat_for_client_auth(request: Request, query: ClientChatQuery, user: dict = Depends(require_auth_client)):
    """認証されたクライアント用チャットエンドポイント"""
    if not core_settings.settings_manager:
        raise HTTPException(503, "Settings manager not initialized")
    
    logging.info(f"Chat request from user: {user.get('email', 'N/A')}")

    # 学生用はデフォルト設定を確実に指定
    default_model = "gemini-2.0-flash" 
    default_embedding = "models/text-embedding-004"

    chat_query = ChatQuery(
        query=query.query,
        model=core_settings.settings_manager.settings.get("model", default_model),
        embedding_model=core_settings.settings_manager.settings.get("embedding_model", default_embedding),
        top_k=core_settings.settings_manager.settings.get("top_k", 5),
        collection=core_settings.settings_manager.settings.get("collection", ACTIVE_COLLECTION_NAME)
    )
    
    return StreamingResponse(enhanced_chat_logic(request, chat_query), media_type="text/event-stream")

@client_router.get("/history")
async def get_chat_history(request: Request, user: dict = Depends(require_auth_client)):
    session_id = get_or_create_session_id(request)
    history = history_manager.get_history(session_id)
    return {"history": history}

@client_router.delete("/history")
async def delete_chat_history(request: Request, user: dict = Depends(require_auth_client)):
    session_id = get_or_create_session_id(request)
    history_manager.clear_history(session_id)
    return {"message": "履歴をクリアしました"}

@router.get("/config")
def get_client_config():
    # 念のためキーがあるか確認
    if not hasattr(config, "SUPABASE_ANON_KEY") or not config.SUPABASE_ANON_KEY:
        # 環境変数が読み込めていない場合の安全策
        raise HTTPException(status_code=500, detail="Server Config Error: ANON KEY not found.")

    return {
        "supabase_url": config.SUPABASE_URL,
        "supabase_anon_key": config.SUPABASE_ANON_KEY
    }