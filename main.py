import os
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from core.config import APP_SECRET_KEY, SUPABASE_URL, SUPABASE_KEY
from core.database import SupabaseClientManager
from core.settings import SettingsManager
from services.document_processor import SimpleDocumentProcessor
# 認証関数
from core.dependencies import require_auth, require_auth_client
# DB変数を格納するモジュールそのものをインポート
from core import database

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
# ★重要: グローバルヘルスチェック (認証なし)
# 他のルーターより先に定義することで、干渉を防ぎます
# =========================================================
@app.get("/health")
def global_health_check():
    # database.db_client を参照することで、最新の状態を確認できます
    status = "supabase" if database.db_client else "uninitialized"
    return {"status": "ok", "database": status}

@app.get("/config")
def get_config():
    return {
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": "YOUR_ANON_KEY_HERE" # 必要に応じて環境変数を使用
    }

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

# 3. 管理者用 API (Admin) - ここには require_auth (認証) がかかります
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