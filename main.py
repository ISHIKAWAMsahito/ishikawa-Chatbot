import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

# coreモジュールのインポート (AI_CONTEXT: os.getenv は core.config の定数を使用)
from core.database import db_client
from core import settings as core_settings
from core.config import SECRET_KEY, APP_SECRET_KEY, IS_PRODUCTION, GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY, PORT
from core.settings import SettingsManager
from core.ws_auth import validate_ws_token

# APIルーターのインポート
from api import chat, feedback, system, auth, documents, fallbacks

# .env ファイルの読み込み
load_dotenv()

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理 (Fail Fast: 本番で必須設定欠落時は起動停止)"""
    logger.info("🚀 Starting up University Support AI...")

    # 0. 本番環境では必須環境変数を厳格チェック (Fail Fast)
    if IS_PRODUCTION:
        if not APP_SECRET_KEY:
            logger.error("❌ APP_SECRET_KEY must be set in production (RENDER). Aborting.")
            raise ValueError("APP_SECRET_KEY must be set in production.")
        if not GEMINI_API_KEY:
            logger.error("❌ GEMINI_API_KEY must be set in production. Aborting.")
            raise ValueError("GEMINI_API_KEY must be set in production.")
        if not SUPABASE_URL:
            logger.error("❌ SUPABASE_URL must be set in production. Aborting.")
            raise ValueError("SUPABASE_URL must be set in production.")
        if not SUPABASE_SERVICE_KEY:
            logger.error("❌ SUPABASE_SERVICE_KEY must be set in production. Aborting.")
            raise ValueError("SUPABASE_SERVICE_KEY must be set in production.")

    # 1. Supabaseクライアントの初期化確認
    if db_client.client:
        logger.info("✅ Supabase client initialized successfully.")
    else:
        logger.error("⚠️ Supabase client is NOT initialized. Check your SUPABASE_URL and KEY.")

    # 2. SettingsManager の初期化
    try:
        core_settings.settings_manager = SettingsManager()
        logger.info("✅ Settings Manager initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Settings Manager: {e}", exc_info=True)
        raise

    yield

    logger.info("👋 Shutting down...")

app = FastAPI(
    title="University Support AI",
    description="RAG-based AI Chatbot for University Students",
    version="2.0.0",
    lifespan=lifespan
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# セッション (core.config の定数を使用、本番では APP_SECRET_KEY 必須)
# ---------------------------------------------------------
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=bool(IS_PRODUCTION),
    same_site="lax",
)

# 静的ファイルの配信
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# ---------------------------------------------------------
# ルーターの登録
# ---------------------------------------------------------

# Chat API (学生用: /api/client/chat/chat となります)
app.include_router(chat.router, prefix="/api/client", tags=["Chat"])

# ★追加: Admin Chat API (管理者用: /api/admin/chat となります)
# admin.html は "/api/admin/chat" にアクセスするため、この登録が必要です
app.include_router(chat.router, prefix="/api/admin", tags=["Admin Chat"])
# Fallbacks API (管理者用: /api/admin/fallbacks に対応)
app.include_router(
    fallbacks.router, 
    prefix="/api/admin/fallbacks", 
    tags=["Admin Fallbacks"]
)
# System API
app.include_router(system.router, prefix="/api/admin/system", tags=["System"])

# Feedback API
app.include_router(feedback.router, prefix="/api", tags=["Feedback"])

# ★追加: Documents API (エラーログ /api/admin/documents/... に対応)
app.include_router(documents.router, prefix="/api/admin/documents", tags=["Documents"])

# Authルーター (HTML配信含むため prefixなし)
app.include_router(auth.router, tags=["Auth"])

# ---------------------------------------------------------
# WebSocket エンドポイント (設定同期用・管理者認証必須)
# ---------------------------------------------------------
@app.websocket("/ws/settings")
async def websocket_settings(websocket: WebSocket):
    """設定画面(admin.html)とのリアルタイム通信用WebSocket。?token=xxx で管理者トークン必須。"""
    token = websocket.query_params.get("token")
    if not validate_ws_token(token):
        logger.warning("WebSocket /ws/settings: 無効または期限切れのトークンで拒否")
        await websocket.close(code=1008)
        return

    if not core_settings.settings_manager:
        logger.error("❌ Settings manager is STILL not initialized.")
        await websocket.close(code=1000)
        return

    try:
        await core_settings.settings_manager.add_websocket(websocket)
        logger.info("✅ WebSocket client connected.")
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        # settings.py のメソッド名 'remove_websocket' を使用
        if core_settings.settings_manager:
            core_settings.settings_manager.remove_websocket(websocket)
        logger.info("WebSocket settings client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if core_settings.settings_manager:
            core_settings.settings_manager.remove_websocket(websocket)

# main.py

# ---------------------------------------------------------
# ヘルスチェック
# ---------------------------------------------------------
@app.get("/health")
def health_check():
    """
    Render用ヘルスチェックおよび管理画面用ステータス確認
    'database' キーを返すことで管理画面の「不明」表示を解消します。
    """
    return {
        "status": "ok",
        # db_client.client が存在すれば "supabase" という文字列を返します
        "database": "supabase" if db_client.client else "uninitialized"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)