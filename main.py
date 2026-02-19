import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel

from core.database import db_client
from core import settings as core_settings
from core.config import (
    SECRET_KEY, IS_PRODUCTION,
    GEMINI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY, PORT,
    ALLOWED_HOSTS,
)
from core.settings import SettingsManager
from core.ws_auth import validate_ws_token

from api import chat, feedback, system, auth, documents, fallbacks, stats

load_dotenv()

# ─────────────────────────────────────────
# ログ設定
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class _HealthCheckFilter(logging.Filter):
    """
    /health, /healthz へのアクセスログを抑制するフィルター。
    Render が5秒ごとに送るヘルスチェックリクエストのノイズを除去する。
    """
    _SUPPRESS = ("/health", "/healthz")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self._SUPPRESS)


# uvicorn のアクセスログにフィルターを適用
for _logger_name in ("uvicorn.access", "uvicorn"):
    logging.getLogger(_logger_name).addFilter(_HealthCheckFilter())


# ─────────────────────────────────────────
# アプリ本体
# ─────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    database: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting up University Support AI...")

    if IS_PRODUCTION:
        missing_vars = []
        if not SECRET_KEY or SECRET_KEY == "default-insecure-key":
            missing_vars.append("APP_SECRET_KEY")
        if not GEMINI_API_KEY:
            missing_vars.append("GEMINI_API_KEY")
        if not SUPABASE_URL:
            missing_vars.append("SUPABASE_URL")
        if not SUPABASE_SERVICE_KEY:
            missing_vars.append("SUPABASE_SERVICE_KEY")

        if missing_vars:
            error_msg = f"❌ CRITICAL: Missing environment variables: {', '.join(missing_vars)}"
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
    lifespan=lifespan,
)

# ─────────────────────────────────────────
# ミドルウェア
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_HOSTS if IS_PRODUCTION else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    https_only=IS_PRODUCTION,
    same_site="lax",
)

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# ─────────────────────────────────────────
# ルーター登録
# ─────────────────────────────────────────
app.include_router(chat.router, prefix="/api/client", tags=["Chat"])
app.include_router(chat.router, prefix="/api/admin", tags=["Admin Chat"])
app.include_router(fallbacks.router, prefix="/api/admin/fallbacks", tags=["Admin Fallbacks"])
app.include_router(system.router, prefix="/api/admin/system", tags=["System"])
app.include_router(documents.router, prefix="/api/admin/documents", tags=["Documents"])
app.include_router(stats.router, prefix="/api/admin/stats", tags=["Admin Stats"])
app.include_router(feedback.router, prefix="/api", tags=["Feedback"])
app.include_router(auth.router, tags=["Auth"])

# ─────────────────────────────────────────
# 静的ページ
# ─────────────────────────────────────────
@app.get("/stats.html")
async def get_stats_page():
    file_path = "static/stats.html"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Stats page not found")
    return FileResponse(file_path)

# ─────────────────────────────────────────
# WebSocket
# ─────────────────────────────────────────
@app.websocket("/ws/settings")
async def websocket_settings(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not validate_ws_token(token):
        logger.warning("WebSocket[Admin] rejected: invalid token")
        await websocket.close(code=1008)
        return

    if not core_settings.settings_manager:
        await websocket.close(code=1000)
        return

    try:
        await core_settings.settings_manager.add_websocket(websocket, is_admin=True)
        logger.info("✅ Admin WebSocket connected.")
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if core_settings.settings_manager:
            core_settings.settings_manager.remove_websocket(websocket)
    except Exception as e:
        logger.error(f"Admin WebSocket error: {e}")
        if core_settings.settings_manager:
            core_settings.settings_manager.remove_websocket(websocket)


@app.websocket("/ws/client/settings")
async def websocket_client_settings(websocket: WebSocket):
    """学生画面用の読み取り専用WebSocket（認証不要）"""
    if not core_settings.settings_manager:
        await websocket.close(code=1000)
        return

    try:
        await core_settings.settings_manager.add_websocket(websocket, is_admin=False)
        logger.info("✅ Client WebSocket connected (Read-only).")
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if core_settings.settings_manager:
            core_settings.settings_manager.remove_websocket(websocket)
    except Exception as e:
        logger.error(f"Client WebSocket error: {e}")
        if core_settings.settings_manager:
            core_settings.settings_manager.remove_websocket(websocket)


# ─────────────────────────────────────────
# ヘルスチェック
# ─────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        database="supabase" if db_client.client else "uninitialized",
    )


@app.api_route("/healthz", methods=["GET", "HEAD"])
def health_check_k8s():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        proxy_headers=True,
        access_log=True,
    )