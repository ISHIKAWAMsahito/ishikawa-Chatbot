import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # ★追加: WebSocket関連
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

# coreモジュールのインポート
from core.database import db_client
from core import settings as core_settings  # ★追加: 設定マネージャー用

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
    """アプリケーションのライフサイクル管理"""
    logger.info("🚀 Starting up University Support AI...")
    
    # Supabaseクライアントの初期化確認
    if db_client.client:
        logger.info("✅ Supabase client initialized successfully.")
    else:
        logger.error("⚠️ Supabase client is NOT initialized. Check your SUPABASE_URL and KEY.")

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

# セッションミドルウェア（ログイン機能に必須）
secret_key = os.getenv("SECRET_KEY", "change-this-to-a-secure-random-string-in-production")
app.add_middleware(SessionMiddleware, secret_key=secret_key)

# 静的ファイルの配信
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# ---------------------------------------------------------
# ルーターの登録
# ---------------------------------------------------------

# API系ルーター
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(feedback.router, prefix="/api", tags=["Feedback"])

# systemルーター（HTTPエンドポイント用）
# /api/health や /api/config などを提供します
app.include_router(system.router, prefix="/api", tags=["System"])

# Authルーター（ログイン・HTML配信）
# /login, /logout, /admin などを提供するため prefixなし
app.include_router(auth.router, tags=["Auth"])

# ---------------------------------------------------------
# WebSocket エンドポイント (設定同期用)
# ---------------------------------------------------------
# system.py から移動されたコードです。
# フロントエンドは "wss://.../ws/settings" に接続しに来ます。

@app.websocket("/ws/settings")
async def websocket_settings(websocket: WebSocket):
    """
    設定画面(admin.html)とのリアルタイム通信用WebSocket
    設定が変更された際に、接続している全クライアントに通知を送るなどの処理に使用
    """
    # SettingsManagerが初期化されているか確認
    if not core_settings.settings_manager:
        logger.error("Settings manager is not initialized.")
        await websocket.close(code=1000)
        return

    try:
        # 接続確立とマネージャーへの登録
        await core_settings.settings_manager.connect(websocket)
        
        # クライアントからのメッセージを待機し続けるループ
        while True:
            # 基本的にサーバーからプッシュ通知を送る用途だが、
            # 切断検知のために receive_text を待つ必要がある
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        # 切断時のクリーンアップ
        core_settings.settings_manager.disconnect(websocket)
        logger.info("WebSocket settings client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        core_settings.settings_manager.disconnect(websocket)

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