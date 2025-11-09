import logging
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from core.dependencies import require_auth_client

# ↓↓↓ [修正]
# from core.settings import settings_manager  <-- バグの原因
from core import settings as core_settings  # <-- "as" で別名を付ける
# ↑↑↑ [修正]

from core.config import ACTIVE_COLLECTION_NAME
from models.schemas import ChatQuery, ClientChatQuery

# ↓↓↓ [ImportErrorの修正]
from services.chat_logic import (
    enhanced_chat_logic, 
    get_or_create_session_id, 
    history_manager
)
# ↑↑↑ [ImportErrorの修正]

router = APIRouter()

@router.post("/chat")
async def chat_endpoint(request: Request, query: ChatQuery):
    """管理者用チャットエンドポイント"""
    # この関数は enhanced_chat_logic に依存
    # enhanced_chat_logic 側（services/chat_logic.py）でも
    # 同様の "core." 修正が必要な可能性が高い
    return StreamingResponse(enhanced_chat_logic(request, query), media_type="text/event-stream")

@router.post("/chat_for_client")
async def chat_for_client_auth(request: Request, query: ClientChatQuery, user: dict = Depends(require_auth_client)):
    """認証されたクライアント用チャットエンドポイント"""
    
    # ↓↓↓ [修正] "core_settings." を使う
    if not core_settings.settings_manager:
        raise HTTPException(503, "Settings manager not initialized")
    # ↑↑↑ [修正]

    logging.info(f"Chat request from user: {user.get('email', 'N/A')}")

    # ↓↓↓ [修正] すべて "core_settings." を使う
    chat_query = ChatQuery(
        query=query.query,
        model=core_settings.settings_manager.settings.get("model", "gemini-2.5-flash"),
        embedding_model=core_settings.settings_manager.settings.get("embedding_model", "text-embedding-004"),
        top_k=core_settings.settings_manager.settings.get("top_k", 5),
        collection=core_settings.settings_manager.settings.get("collection", ACTIVE_COLLECTION_NAME)
    )
    # ↑↑↑ [修正]
    
    return StreamingResponse(enhanced_chat_logic(request, chat_query), media_type="text/event-stream")

@router.get("/chat/history")
async def get_chat_history(request: Request, user: dict = Depends(require_auth_client)):
    """現在のセッションのチャット履歴を取得"""
    session_id = get_or_create_session_id(request)
    
    # ↓↓↓ [ImportErrorの修正] 呼び出し方を変更
    history = history_manager.get_history(session_id)
    # ↑↑↑ [ImportErrorの修正]
    
    return {"history": history}

@router.delete("/chat/history")
async def delete_chat_history(request: Request, user: dict = Depends(require_auth_client)):
    """現在のセッションのチャット履歴を削除"""
    session_id = get_or_create_session_id(request)
    
    # ↓↓↓ [ImportErrorの修正] 呼び出し方を変更
    history_manager.clear_history(session_id)
    # ↑↑↑ [ImportErrorの修正]
    
    return {"message": "履歴をクリアしました"}