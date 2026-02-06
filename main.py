import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 修正ポイント1: 存在しない SupabaseClientManager を削除し、db_client をインポート
from core.database import db_client
from core.constants import PARAMS

# APIルーターのインポート（プロジェクト構成に合わせて調整してください）
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
    """
    アプリケーションの起動・終了時のライフサイクル管理
    """
    logger.info("🚀 Starting up University Support AI...")

    # 修正ポイント2: db_client を使用して接続状態をログ出力
    # core/database.py で既に初期化されているため、ここでは確認のみ行います
    if db_client.client:
        logger.info("✅ Supabase client initialized successfully.")
    else:
        logger.warning("⚠️ Supabase client is NOT initialized. Check your SUPABASE_URL and KEY.")

    yield
    
    logger.info("👋 Shutting down...")

app = FastAPI(
    title="University Support AI",
    description="RAG-based AI Chatbot for University Students",
    version="2.0.0",
    lifespan=lifespan
)

# CORS設定（フロントエンドからのアクセス許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では具体的なドメイン（例: ["https://myapp.onrender.com"]）を指定推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイルの配信設定（staticディレクトリが存在する場合のみ）
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# APIルーターの登録
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(feedback.router, prefix="/api", tags=["Feedback"])

# 以下のルーターは必要に応じてコメントアウトを解除してください
# app.include_router(auth.router, prefix="/api", tags=["Auth"])
# app.include_router(documents.router, prefix="/api", tags=["Documents"])
# app.include_router(system.router, prefix="/api", tags=["System"])
# app.include_router(fallbacks.router, prefix="/api", tags=["Fallbacks"])

@app.get("/")
def read_root():
    """ヘルスチェック用エンドポイント"""
    return {"status": "ok", "message": "University Support AI is running."}

if __name__ == "__main__":
    import uvicorn
    # Renderなどの環境変数 PORT に対応
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)