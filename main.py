import os
import logging
import uvicorn
from contextlib import asynccontextmanager

# 必要なモジュールをすべてインポート (Dependsを含む)
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from core.config import APP_SECRET_KEY, SUPABASE_URL, SUPABASE_KEY
from core.database import SupabaseClientManager
from core.settings import SettingsManager
from services.document_processor import SimpleDocumentProcessor
from core import database
from core import settings as core_settings
from core.dependencies import require_auth

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

# セッション設定 (Brave対策: same_site='lax'に変更)
app.add_middleware(
    SessionMiddleware, 
    secret_key=APP_SECRET_KEY,
    https_only=True,
    same_site='lax'
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

@app.get("/DB.html", dependencies=[Depends(require_auth)])
async def get_db_page():
    # main.py の場所を基準に、DB.html の絶対パスを計算
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "DB.html")
    
    # ログにパスを出力しておくと、デバッグ時に役立ちます（Renderのログで確認できます）
    logging.info(f"Trying to serve file from: {file_path}")

    if not os.path.exists(file_path):
        # ファイルがない場合は明確にエラーを出す
        logging.error(f"File NOT found at: {file_path}")
        raise RuntimeError(f"File at path {file_path} does not exist.")
        
    return FileResponse(file_path)
# =========================================================
# WebSocketエンドポイント
# =========================================================
@app.websocket("/ws/settings")
async def websocket_endpoint(websocket: WebSocket):
    """設定変更通知用WebSocket"""
    if not core_settings.settings_manager:
        await websocket.close(code=1011, reason="Settings manager not initialized")
        return
    
    await core_settings.settings_manager.add_websocket(websocket)
    
    try:
        current_settings = core_settings.settings_manager.settings
        await websocket.send_json({
            "type": "settings_update",
            "data": current_settings
        })
    except Exception as e:
        logging.error(f"WebSocketへの初期設定送信に失敗: {e}")

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        core_settings.settings_manager.remove_websocket(websocket)

# ---------------------------------------------------------
# ルーター登録
# ---------------------------------------------------------

# 静的ファイルのmountはAPIルーターより前に行う
app.mount("/static", StaticFiles(directory="static"), name="static")

# Authルーター (HTML配信もここで担当)
app.include_router(auth.router, tags=["Auth"])

# APIルーター群
app.include_router(chat.router, prefix="/api/client/chat", tags=["Client Chat"])
app.include_router(feedback.router, prefix="/api/client/feedback", tags=["Client Feedback"])

# 管理者API (APIエンドポイントは厳密に保護)
app.include_router(documents.router, prefix="/api/admin/documents", tags=["Admin Documents"], dependencies=[Depends(require_auth)])
app.include_router(fallbacks.router, prefix="/api/admin/fallbacks", tags=["Admin Fallbacks"], dependencies=[Depends(require_auth)])
app.include_router(system.router, prefix="/api/admin/system", tags=["Admin System"], dependencies=[Depends(require_auth)])
app.include_router(chat.router, prefix="/api/admin/chat", tags=["Admin Chat"], dependencies=[Depends(require_auth)])

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)