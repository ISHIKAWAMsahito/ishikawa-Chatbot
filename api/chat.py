import logging
from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from core.dependencies import require_auth_client
from core import settings as core_settings
from core.config import ACTIVE_COLLECTION_NAME, SUPABASE_URL, SUPABASE_ANON_KEY
from models.schemas import ChatQuery, ClientChatQuery
from services.chat_logic import enhanced_chat_logic, get_or_create_session_id, history_manager

# ★重要: ルーターを2つ定義
admin_router = APIRouter()   # 管理者用
client_router = APIRouter()  # 学生用

# ============================
# 管理者用エンドポイント (admin_router)
# ============================
@admin_router.post("/")
async def chat_endpoint(request: Request, query: ChatQuery):
    """管理者用チャットエンドポイント"""
    return StreamingResponse(enhanced_chat_logic(request, query), media_type="text/event-stream")

# ============================
# 学生用エンドポイント (client_router)
# ============================
@client_router.post("/")
async def chat_for_client_auth(request: Request, query: ClientChatQuery, user: dict = Depends(require_auth_client)):
    """認証されたクライアント用チャットエンドポイント"""
    if not core_settings.settings_manager:
        raise HTTPException(503, "Settings manager not initialized")
    
    logging.info(f"Chat request from user: {user.get('email', 'N/A')}")

    # 学生用はデフォルト設定を確実に指定
    # ★修正: 3072次元のDBに合わせるため、3072次元対応の 'gemini-embedding-001' を使用
    default_model = "gemini-2.5-flash" 
    default_embedding = "models/gemini-embedding-001"

    chat_query = ChatQuery(
        query=query.query,
        model=core_settings.settings_manager.settings.get("model", default_model),
        embedding_model=core_settings.settings_manager.settings.get("embedding_model", default_embedding),
        top_k=core_settings.settings_manager.settings.get("top_k", 5),
        collection=core_settings.settings_manager.settings.get("collection", ACTIVE_COLLECTION_NAME)
    )
    
    return StreamingResponse(enhanced_chat_logic(request, chat_query), media_type="text/event-stream")

# ★修正: 認証なしで /config にアクセス可能にする
@client_router.get("/config")
async def get_client_config():
    """
    クライアント画面に対して、Supabaseの設定を返す
    （公開用キー: SUPABASE_ANON_KEY のみ）
    ※ 認証なしでアクセス可能（クライアント初期化時に必要）
    """
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Supabase設定が不完全です"
        )
    
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY
    }

@client_router.get("/history")
async def get_chat_history(request: Request, user: dict = Depends(require_auth_client)):
    """現在のセッションのチャット履歴を取得"""
    session_id = get_or_create_session_id(request)
    history = history_manager.get_history(session_id)
    return {"history": history}

@client_router.delete("/history")
async def delete_chat_history(request: Request, user: dict = Depends(require_auth_client)):
    """現在のセッションのチャット履歴を削除"""
    session_id = get_or_create_session_id(request)
    history_manager.clear_history(session_id)
    return {"message": "履歴をクリアしました"}