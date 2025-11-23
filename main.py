import os
import logging
import uvicorn
from contextlib import asynccontextmanager
# ↓↓↓ WebSocket関連をインポート
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from core.config import APP_SECRET_KEY, SUPABASE_URL, SUPABASE_KEY
from core.database import SupabaseClientManager
from core.settings import SettingsManager
from services.document_processor import SimpleDocumentProcessor
from core.dependencies import require_auth, require_auth_client
from core import database
# ↓↓↓ 設定モジュールをインポート（WebSocketで使用）
from core import settings as core_settings

# APIルーター
from api import auth, chat, documents, fallbacks, feedback, system

# ログ設定
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    logging.info("--- アプリケーション起動 ---")
    
    # 1. 設定マネージャー初期化
    from core import settings as settings_module
    settings_module.settings_manager = SettingsManager()
    
    # 2. ドキュメントプロセッサ初期化
    from services import document_processor
    document_processor.simple_processor = SimpleDocumentProcessor(chunk_size=1000, chunk_overlap=200)

    # 3. Supabase初期化
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            database.db_client = SupabaseClientManager(url=SUPABASE_URL, key=SUPABASE_KEY)
            logging.info("Supabaseクライアント初期化完了")
        except Exception as e:
            logging.error(f"Supabase初期化エラー: {e}")
    else:
        logging.warning("Supabase設定が見つかりません")

    yield
    logging.info("--- アプリケーション終了 ---")

app = FastAPI(lifespan=lifespan)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# セッション設定
app.add_middleware(
    SessionMiddleware, 
    secret_key=APP_SECRET_KEY,
    https_only=True,
    same_site='none'
)

Instrumentator().instrument(app).expose(app)

# =========================================================
# グローバルヘルスチェック (認証なし)
# =========================================================
@app.get("/health")
def global_health_check():
    status = "supabase" if database.db_client else "uninitialized"
    return {"status": "ok", "database": status}

@app.get("/healthz")
def healthz_check():
    return {"status": "ok"}

@app.get("/config")
def get_config():
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": "YOUR_ANON_KEY_HERE" 
    }

# =========================================================
# ★ WebSocketエンドポイント (認証なし・ルート直下)
# =========================================================
@app.websocket("/ws/settings")
async def websocket_endpoint(websocket: WebSocket):
    """設定変更通知用WebSocket"""
    if not core_settings.settings_manager:
        await websocket.close(code=1011, reason="Settings manager not initialized")
        return
    
    await core_settings.settings_manager.add_websocket(websocket)
    
    try:
        # 接続直後に、現在の設定をこのクライアントに送信する
        current_settings = core_settings.settings_manager.settings
        await websocket.send_json({
            "type": "settings_update",
            "data": current_settings
        })
        logging.info(f"WebSocketクライアントに初期設定を送信しました。")
        
    except Exception as e:
        logging.error(f"WebSocketへの初期設定送信に失敗: {e}")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        core_settings.settings_manager.remove_websocket(websocket)
        logging.info("WebSocketクライアントが切断されました。")


# ---------------------------------------------------------
# ルーター登録
# ---------------------------------------------------------

# 1. 認証 (Auth)
app.include_router(auth.router, tags=["Auth"])

# 2. 学生用 API (Client)
app.include_router(
    chat.router,
    prefix="/api/client/chat", 
    tags=["Client Chat"]
)
app.include_router(
    feedback.router,
    prefix="/api/client/feedback",
    tags=["Client Feedback"]
)

# 3. 管理者用 API (Admin)
app.include_router(
    documents.router,
    prefix="/api/admin/documents", 
    tags=["Admin Documents"],
    dependencies=[Depends(require_auth)] 
)
app.include_router(
    fallbacks.router,
    prefix="/api/admin/fallbacks", 
    tags=["Admin Fallbacks"],
    dependencies=[Depends(require_auth)]
)
app.include_router(
    system.router,
    prefix="/api/admin/system",
    tags=["Admin System"],
    dependencies=[Depends(require_auth)]
)
app.include_router(
    chat.router,
    prefix="/api/admin/chat",
    tags=["Admin Chat"],
    dependencies=[Depends(require_auth)]
)

# ---------------------------------------------------------
# 静的ファイルとルート設定
# ---------------------------------------------------------

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/client.html")

@app.get("/client.html")
async def client_page():
    return RedirectResponse(url="/static/client.html")

@app.get("/admin.html")
async def admin_page():
    return RedirectResponse(url="/static/admin.html")

@app.get("/DB.html")
async def db_page():
    return RedirectResponse(url="/static/DB.html")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)