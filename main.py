import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

# coreモジュールのインポート
from core.database import db_client
from core import settings as core_settings
# SettingsManagerクラスをインポート
from core.settings import SettingsManager 

# APIルーターのインポート
from api import chat, feedback, system, auth

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
    """アプリケーションのライフサイクル管理"""
    logger.info("🚀 Starting up University Support AI...")
    
    # 1. Supabaseクライアントの初期化確認
    if db_client.client:
        logger.info("✅ Supabase client initialized successfully.")
    else:
        logger.error("⚠️ Supabase client is NOT initialized. Check your SUPABASE_URL and KEY.")

    # 2. SettingsManager の初期化 (★修正箇所)
    try:
        # settings.py の定義に合わせて、引数なしで初期化します
        core_settings.settings_manager = SettingsManager()
        logger.info("✅ Settings Manager initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Settings Manager: {e}", exc_info=True)

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
# 環境変数名の不一致を解消 (Render対応)
# ---------------------------------------------------------
# Renderのスクリーンショットにある "APP_SECRET_KEY" を読み込みます
secret_key = os.getenv("APP_SECRET_KEY") or os.getenv("SECRET_KEY", "default-insecure-key")
is_production = os.getenv("RENDER") is not None

app.add_middleware(
    SessionMiddleware, 
    secret_key=secret_key,
    https_only=is_production, # 本番環境(Render)ではHTTPS必須
    same_site="lax"           # CSRF対策
)

# 静的ファイルの配信
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# ---------------------------------------------------------
# ルーターの登録
# ---------------------------------------------------------

# Chat API
app.include_router(chat.router, prefix="/api/client/chat", tags=["Chat"])

# System API
app.include_router(system.router, prefix="/api/admin/system", tags=["System"])

# Feedback API
app.include_router(feedback.router, prefix="/api", tags=["Feedback"])

# Authルーター (HTML配信含むため prefixなし)
app.include_router(auth.router, tags=["Auth"])

# ---------------------------------------------------------
# WebSocket エンドポイント (設定同期用)
# ---------------------------------------------------------
@app.websocket("/ws/settings")
async def websocket_settings(websocket: WebSocket):
    """設定画面(admin.html)とのリアルタイム通信用WebSocket"""
    
    # 初期化チェック
    if not core_settings.settings_manager:
        logger.error("❌ Settings manager is STILL not initialized.")
        await websocket.close(code=1000)
        return

    try:
        # ★修正: settings.py のメソッド名 'add_websocket' を使用
        await core_settings.settings_manager.add_websocket(websocket)
        logger.info("✅ WebSocket client connected.")
        
        while True:
            # 切断検知のためにメッセージ待ち
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        # ★修正: settings.py のメソッド名 'remove_websocket' を使用
        if core_settings.settings_manager:
            core_settings.settings_manager.remove_websocket(websocket)
        logger.info("WebSocket settings client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if core_settings.settings_manager:
            core_settings.settings_manager.remove_websocket(websocket)

# ---------------------------------------------------------
# ヘルスチェック
# ---------------------------------------------------------
@app.get("/health")
def health_check():
    """Render用ヘルスチェック"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)