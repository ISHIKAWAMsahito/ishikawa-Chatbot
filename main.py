import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel

# coreモジュールのインポート
from core.database import db_client
from core import settings as core_settings
from core.config import (
    SECRET_KEY, APP_SECRET_KEY, IS_PRODUCTION, 
    GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY, PORT,
    ALLOWED_HOSTS
)
from core.settings import SettingsManager
from core.ws_auth import validate_ws_token

# APIルーター
from api import chat, feedback, system, auth, documents, fallbacks

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# ヘルスチェック用レスポンスモデル (Strict Typing)
# ---------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    database: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理 (Fail Fast)"""
    logger.info("🚀 Starting up University Support AI...")

    # 1. 本番環境 (Fail Fast Check)
    if IS_PRODUCTION:
        missing_vars = []
        if not APP_SECRET_KEY: missing_vars.append("APP_SECRET_KEY")
        if not GEMINI_API_KEY: missing_vars.append("GEMINI_API_KEY")
        if not SUPABASE_URL: missing_vars.append("SUPABASE_URL")
        if not SUPABASE_SERVICE_KEY: missing_vars.append("SUPABASE_SERVICE_KEY")
        
        if missing_vars:
            error_msg = f"❌ CRITICAL: Missing environment variables in production: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # 2. Supabase初期化確認
    if db_client.client:
        logger.info("✅ Supabase client initialized.")
    else:
        logger.error("⚠️ Supabase client is NOT initialized.")

    # 3. SettingsManager初期化
    try:
        core_settings.settings_manager = SettingsManager()
        logger.info("✅ Settings Manager initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Settings Manager: {e}", exc_info=True)
        # 設定マネージャは必須のため、失敗時は起動しない選択も可
        raise

    yield

    logger.info("👋 Shutting down...")

app = FastAPI(
    title="University Support AI",
    description="RAG-based AI Chatbot",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_HOSTS + ["*"], # 開発用に*も含めるが、本番は制限推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# セッション
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=IS_PRODUCTION,
    same_site="lax",
)

# 静的ファイル
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# ---------------------------------------------------------
# ルーター登録
# ---------------------------------------------------------

# Chat API
app.include_router(chat.router, prefix="/api/client", tags=["Chat"])
app.include_router(chat.router, prefix="/api/admin", tags=["Admin Chat"])

# Fallbacks API (管理者用)
# fallbacks.py 側で @router.get("") としているため、
# ここでの prefix="/api/admin/fallbacks" がそのままルートパスになります
app.include_router(
    fallbacks.router, 
    prefix="/api/admin/fallbacks", 
    tags=["Admin Fallbacks"]
)

# System API
app.include_router(system.router, prefix="/api/admin/system", tags=["System"])
# Documents API
app.include_router(documents.router, prefix="/api/admin/documents", tags=["Documents"])
# Feedback API
app.include_router(feedback.router, prefix="/api", tags=["Feedback"])
# Auth
app.include_router(auth.router, tags=["Auth"])

# ---------------------------------------------------------
# WebSocket
# ---------------------------------------------------------
app.websocket("/ws/settings")
async def websocket_settings(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not validate_ws_token(token):
        logger.warning("WebSocket /ws/settings: Invalid or expired token.")
        await websocket.close(code=1008)
        return

    if not core_settings.settings_manager:
        await websocket.close(code=1000)
        return

    try:
        await core_settings.settings_manager.add_websocket(websocket)
        logger.info("✅ WS Client connected.")
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        core_settings.settings_manager.remove_websocket(websocket)
    except Exception:
        core_settings.settings_manager.remove_websocket(websocket)

# ---------------------------------------------------------
# ヘルスチェック
# ---------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Render用ヘルスチェック"""
    return HealthResponse(
        status="ok",
        database="supabase" if db_client.client else "uninitialized"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)