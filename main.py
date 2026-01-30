import os
import logging
import uvicorn
import asyncio
from contextlib import asynccontextmanager

# 必要なモジュールをインポート
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, Request, status, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse
from starlette.middleware.sessions import SessionMiddleware
from fastapi.middleware.cors import CORSMiddleware

# Core & Config
# ★重要: ここで config をインポートすることで、環境変数のロードとLangSmith設定が完了します
from core import config 
from core.config import APP_SECRET_KEY, SUPABASE_URL, SUPABASE_KEY, SUPABASE_ANON_KEY
from core.database import SupabaseClientManager
from core.settings import SettingsManager
from services.document_processor import SimpleDocumentProcessor
from core import database
from core import settings as core_settings
from core.dependencies import require_auth

# APIルーター
from api import auth, chat, documents, fallbacks, feedback, system

# --- 環境判定 ---
ENV_TYPE = os.getenv("ENVIRONMENT", "production")
IS_LOCAL = ENV_TYPE == "local"

# ログ設定
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    logging.info(f"--- アプリケーション起動 (モード: {ENV_TYPE}) ---")
    
    # 0. LangSmith 接続確認
    if config.LANGCHAIN_TRACING_V2:
        logging.info(f"✅ LangSmith Tracing: ON (Project: {config.LANGCHAIN_PROJECT})")
    else:
        logging.info("🚫 LangSmith Tracing: OFF")

    # 1. 設定マネージャー初期化
    from core import settings as settings_module
    settings_module.settings_manager = SettingsManager()
    logging.info("✅ 設定マネージャー初期化完了")
    
    # 2. ドキュメントプロセッサ初期化
    from services import document_processor
    document_processor.simple_processor = SimpleDocumentProcessor(
        parent_chunk_size=1500,
        child_chunk_size=400
    )
    logging.info("✅ ドキュメントプロセッサ初期化完了 (Parent-Child Chunking)")

    # 3. Supabase初期化
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            logging.info("⏳ Supabase初期化中...")
            def init_supabase():
                return SupabaseClientManager(url=SUPABASE_URL, key=SUPABASE_KEY)
            try:
                database.db_client = await asyncio.wait_for(
                    asyncio.to_thread(init_supabase),
                    timeout=10.0
                )
                logging.info("✅ Supabaseクライアント初期化完了")
            except asyncio.TimeoutError:
                logging.warning("⚠️ Supabase初期化タイムアウト（起動は続行）")
                database.db_client = None
        except Exception as e:
            logging.error(f"❌ Supabase初期化エラー: {e}")
            database.db_client = None
    else:
        logging.warning("⚠️ Supabase設定が見つかりません")

    yield
    logging.info("--- アプリケーション終了 ---")

app = FastAPI(lifespan=lifespan)

# --- CORS設定 ---
allowed_origins = ["http://localhost:8000", "http://127.0.0.1:8000"]
if not IS_LOCAL:
    prod_url = os.getenv("APP_URL")
    if prod_url:
        allowed_origins.append(prod_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- セッション設定 ---
app.add_middleware(
    SessionMiddleware,
    secret_key=APP_SECRET_KEY,
    https_only=not IS_LOCAL,
    same_site='lax'
)


# =========================================================
# グローバルヘルスチェック
# =========================================================
@app.get("/health")
def global_health_check():
    status = "supabase" if database.db_client else "uninitialized"
    return {
        "status": "ok", 
        "database": status, 
        "mode": ENV_TYPE,
        "tracing": "enabled" if config.LANGCHAIN_TRACING_V2 else "disabled"
    }

@app.get("/healthz")
def healthz_check():
    return {"status": "ok"}


# =========================================================
# HTMLファイルのルーティング
# =========================================================
@app.get("/stats.html")
async def get_stats_page():
    return FileResponse("static/stats.html")

@app.get("/DB.html")
async def get_db_page():
    return FileResponse("static/DB.html")

@app.get("/admin.html")
async def get_admin_page(user: dict = Depends(require_auth)):
    return FileResponse("static/admin.html")


@app.get("/api/admin/stats/data", dependencies=[Depends(require_auth)])
async def get_admin_stats_data():
    if not database.db_client:
        logging.error("❌ Database client is not initialized")
        raise HTTPException(status_code=503, detail="Database not initialized")
    try:
        client = database.db_client
        if hasattr(client, "client"):
            client = client.client
        logging.info("⏳ Admin Stats: Fetching data from Supabase...")

        response = client.table("anonymous_comments")\
            .select("*")\
            .order("created_at", desc=True)\
            .limit(100)\
            .execute()
        logging.info(f"✅ Data fetched successfully: {len(response.data)} items")
        return response.data

    except Exception as e:
        logging.error(f"❌ Stats fetch error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")


# =========================================================
# WebSocketエンドポイント
# =========================================================
@app.websocket("/ws/settings")
async def websocket_endpoint(websocket: WebSocket):
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
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(auth.router, tags=["Auth"])
app.include_router(chat.client_router, prefix="/api/client/chat", tags=["Client Chat"])
app.include_router(feedback.router, prefix="/api/client/feedback", tags=["Client Feedback"])

# 管理者API
app.include_router(documents.router, prefix="/api/admin/documents", tags=["Admin Documents"], dependencies=[Depends(require_auth)])
app.include_router(fallbacks.router, prefix="/api/admin/fallbacks", tags=["Admin Fallbacks"], dependencies=[Depends(require_auth)])
app.include_router(system.router, prefix="/api/admin/system", tags=["Admin System"], dependencies=[Depends(require_auth)])
app.include_router(chat.admin_router, prefix="/api/admin/chat", tags=["Admin Chat"], dependencies=[Depends(require_auth)])

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    # proxy_headers=True, forwarded_allow_ips="*" はRender等のリバースプロキシ環境で有用
    uvicorn.run("main:app", host="0.0.0.0", port=port, proxy_headers=True, forwarded_allow_ips="*")