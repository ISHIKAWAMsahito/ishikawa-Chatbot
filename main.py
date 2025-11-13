import os
import logging
import uvicorn
# ログ設定
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')
from contextlib import asynccontextmanager
from fastapi import FastAPI
from starlette.middleware.sessions import SessionMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from core.config import APP_SECRET_KEY, SUPABASE_URL, SUPABASE_KEY
from core.database import db_client, SupabaseClientManager
from core.settings import settings_manager, SettingsManager
from services.document_processor import simple_processor, SimpleDocumentProcessor

# APIルーターのインポート
from api import auth, chat, documents, fallbacks, feedback, system



# ライフサイクル管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    # グローバル変数を参照するためにimport時の変数を使用
    from core import database, settings as settings_module
    from services import document_processor
    
    logging.info("--- アプリケーション起動処理開始 ---")

    # SettingsManager初期化
    settings_module.settings_manager = SettingsManager()
    
    # SimpleDocumentProcessor初期化
    document_processor.simple_processor = SimpleDocumentProcessor(chunk_size=1000, chunk_overlap=200)

    # Supabaseクライアント初期化
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            database.db_client = SupabaseClientManager(url=SUPABASE_URL, key=SUPABASE_KEY)
            logging.info("Supabaseクライアントの初期化完了。")
        except Exception as e:
            # ↓↓↓↓ ここから修正 ↓↓↓↓
            error_message = f"Supabase初期化エラー: {e}"
            logging.error(error_message)
            # ログが隠蔽されても、これで強制的にクラッシュさせてエラーを表示する
            raise RuntimeError(error_message)
    else:
        logging.warning("Supabaseの環境変数が設定されていません。")

    yield

    logging.info("--- アプリケーション終了処理 ---")

# FastAPIアプリケーション作成
# 1. FastAPIアプリケーション作成
app = FastAPI(lifespan=lifespan)

# 2. app を定義した後に middleware を追加
app.add_middleware(
    SessionMiddleware, 
    secret_key=APP_SECRET_KEY,
    https_only=True,   # ⬅️ ★ 'secure=True' ではなく、こちらを使います
    same_site='none'   # ⬅️ 'wss://' (WebSocket) 接続を許可します
)

# Prometheusメトリクス
Instrumentator().instrument(app).expose(app)

# ルーターの登録
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(fallbacks.router)
app.include_router(feedback.router)
app.include_router(system.router)

# 開発用サーバー起動
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)