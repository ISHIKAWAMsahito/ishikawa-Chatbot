import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
# ★追加: セッション管理用ミドルウェア
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv

# core.database から db_client をインポート
from core.database import db_client
from core.constants import PARAMS

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
    """
    アプリケーションの起動・終了時のライフサイクル管理
    """
    logger.info("🚀 Starting up University Support AI...")

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

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ★追加: セッションミドルウェアの設定
# これがないと auth.py の request.session['user'] でエラーになります
# SECRET_KEYは.envに設定するか、なければランダムな文字列を使用
secret_key = os.getenv("SECRET_KEY", "your-very-secret-key-change-in-production")
app.add_middleware(SessionMiddleware, secret_key=secret_key)

# 静的ファイルの配信
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# APIルーターの登録
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(feedback.router, prefix="/api", tags=["Feedback"])

# ★追加: 認証用ルーターの登録
# auth.py はHTML配信も兼ねているため、prefix="/api" は付けずにルートにマウントします
# これにより /login, /logout, /admin などが有効になります
app.include_router(auth.router, tags=["Auth"])

# ヘルスチェック用エンドポイント
# ★修正: auth.router にもルートパス("/")の定義(client.html配信)があるため、
# ここの "/" は削除するか、認証が不要なAPI専用のヘルスチェックとして別名にします。
# 今回はRender用のヘルスチェック "/health" が既にあるため、競合する "/" は削除しました。

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)