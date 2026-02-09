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
    SECRET_KEY, IS_PRODUCTION, 
    GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY, PORT,
    ALLOWED_HOSTS
)
from core.settings import SettingsManager
from core.ws_auth import validate_ws_token

# APIルーター
from api import chat, feedback, system, auth, documents, fallbacks

load_dotenv()

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class HealthResponse(BaseModel):
    status: str
    database: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    logger.info("🚀 Starting up University Support AI...")

    if IS_PRODUCTION:
        missing_vars = []
        if not SECRET_KEY or SECRET_KEY == "default-insecure-key": missing_vars.append("APP_SECRET_KEY")
        if not GEMINI_API_KEY: missing_vars.append("GEMINI_API_KEY")
        if not SUPABASE_URL: missing_vars.append("SUPABASE_URL")
        if not SUPABASE_SERVICE_KEY: missing_vars.append("SUPABASE_SERVICE_KEY")
        
        if missing_vars:
            error_msg = f"❌ CRITICAL: Missing environment variables in production: {', '.join(missing_vars)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    if db_client.client:
        logger.info("✅ Supabase client initialized.")
    else:
        logger.error("⚠️ Supabase client is NOT initialized.")

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
    description="RAG-based AI Chatbot",
    version="2.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------
# ミドルウェア設定
# ---------------------------------------------------------

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_HOSTS if IS_PRODUCTION else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# セッション: プロキシ環境(Render)での Cookie 処理を安定化
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=IS_PRODUCTION, # 本番はHTTPS必須
    same_site="lax",          # リダイレクト時のCookie維持のため lax
)

# 静的ファイル
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# ---------------------------------------------------------
# ルーター登録
# ---------------------------------------------------------
app.include_router(chat.router, prefix="/api/client", tags=["Chat"])
app.include_router(chat.router, prefix="/api/admin", tags=["Admin Chat"])
app.include_router(fallbacks.router, prefix="/api/admin/fallbacks", tags=["Admin Fallbacks"])
app.include_router(system.router, prefix="/api/admin/system", tags=["System"])
app.include_router(documents.router, prefix="/api/admin/documents", tags=["Documents"])
app.include_router(feedback.router, prefix="/api", tags=["Feedback"])
app.include_router(auth.router, tags=["Auth"])

# ---------------------------------------------------------
# WebSocket (トークン必須)
# ---------------------------------------------------------
@app.websocket("/ws/settings")
async def websocket_settings(websocket: WebSocket):
    token = websocket.query_params.get("token")
    
    # 詳細なログ出力で接続拒否の原因を特定
    if not validate_ws_token(token):
        masked_token = (token[:5] + "...") if token else "None"
        logger.warning(f"WebSocket auth failed. Token: {masked_token}")
        await websocket.close(code=1008)
        return

    if not core_settings.settings_manager:
        await websocket.close(code=1000)
        return

    try:
        await core_settings.settings_manager.add_websocket(websocket)
        logger.info("✅ WebSocket client connected.")
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if core_settings.settings_manager:
            core_settings.settings_manager.remove_websocket(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if core_settings.settings_manager:
            core_settings.settings_manager.remove_websocket(websocket)

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        database="supabase" if db_client.client else "uninitialized"
    )

if __name__ == "__main__":
    import uvicorn
    # ★重要: proxy_headers=True で Render からの X-Forwarded-Proto を信頼し、
    # アプリが自身を https:// と認識できるようにする (Auth0 エラー対策)
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True, proxy_headers=True)