import os
import logging
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from authlib.integrations.starlette_client import OAuth

# ----------------------------------------------------------------
# 1. 環境変数の読み込み設定
# ----------------------------------------------------------------
# ロギング設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 本番判定: RENDER が設定されていれば本番
IS_PRODUCTION = bool(os.getenv("RENDER"))

if not IS_PRODUCTION:
    # ローカル開発環境: .env を読み込む
    load_dotenv()
    logging.info("✅ ローカル環境: .env から設定を読み込みました。")
else:
    logging.info("🚀 本番環境として起動しました (Renderの環境変数を使用)。")

# ----------------------------------------------------------------
# 2. Gemini API & モデル設定 (指針に基づき厳格固定)
# ----------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Fail Fast: 本番環境でAPIキーがない場合は即停止
if IS_PRODUCTION and not GEMINI_API_KEY:
    raise ValueError("CRITICAL: 'GEMINI_API_KEY' must be set in production environment.")
elif not GEMINI_API_KEY:
    logging.error("⚠️ 環境変数 'GEMINI_API_KEY' が設定されていません。")

# ★ モデル標準化 (AI_CONTEXT 指針 3. Interface & Logic Integrity)
# 生成・リランク用
LLM_MODEL = "models/gemini-2.5-flash"
# 埋め込み用 (004は使用禁止)
EMBEDDING_MODEL = "models/gemini-embedding-001"
# 互換性用エイリアス
EMBEDDING_MODEL_DEFAULT = EMBEDDING_MODEL

# ----------------------------------------------------------------
# 3. LangSmith (LangChain) 設定
# ----------------------------------------------------------------
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ishikawa-chatbot-eval")

if LANGCHAIN_TRACING_V2:
    if not LANGCHAIN_API_KEY:
        logging.warning("⚠️ LangSmith Tracing is enabled but API Key is missing.")
    else:
        logging.info(f"🔎 LangSmith Tracing: ENABLED (Project: {LANGCHAIN_PROJECT})")

# ----------------------------------------------------------------
# 4. APIキー & 認証設定 (Auth0)
# ----------------------------------------------------------------
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")

# セッション秘密鍵の統合ロジック
# 環境変数が APP_SECRET_KEY でも SECRET_KEY でもここで吸収する
raw_secret = os.getenv("APP_SECRET_KEY") or os.getenv("SECRET_KEY")

if IS_PRODUCTION:
    if not raw_secret or raw_secret == "default-insecure-key":
        raise ValueError("CRITICAL: Secure 'APP_SECRET_KEY' is required in production.")
    SECRET_KEY = raw_secret
else:
    if not raw_secret:
        logging.warning("⚠️ 'APP_SECRET_KEY' 未設定。開発用デフォルトキーを使用します。")
        SECRET_KEY = "default-insecure-key"
    else:
        SECRET_KEY = raw_secret

# 互換性のため APP_SECRET_KEY も定義しておく
APP_SECRET_KEY = SECRET_KEY

# オープンリダイレクト対策: 許可するホスト
# Renderのドメインをデフォルトで許可リストに追加
DEFAULT_HOSTS = "localhost,127.0.0.1,ishikawa-chatbot.onrender.com"
ALLOWED_HOSTS_STR = os.getenv("ALLOWED_HOSTS", DEFAULT_HOSTS)
ALLOWED_HOSTS: list[str] = [h.strip().lower() for h in ALLOWED_HOSTS_STR.split(",") if h.strip()]

# ----------------------------------------------------------------
# 5. Supabase設定
# ----------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_KEY = SUPABASE_SERVICE_KEY # エイリアス

if IS_PRODUCTION:
    if not SUPABASE_URL:
        raise ValueError("CRITICAL: 'SUPABASE_URL' is missing in production.")
    if not SUPABASE_SERVICE_KEY:
        raise ValueError("CRITICAL: 'SUPABASE_SERVICE_KEY' is missing in production.")

if not SUPABASE_ANON_KEY:
    logging.warning("⚠️ 'SUPABASE_ANON_KEY' が設定されていません。")

# ----------------------------------------------------------------
# 6. その他定数
# ----------------------------------------------------------------
PORT = int(os.getenv("PORT", "8000"))
ACTIVE_COLLECTION_NAME = "student-knowledge-base"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JST = timezone(timedelta(hours=+9), 'JST')

SUPER_ADMIN_EMAILS = [e.strip() for e in os.getenv("SUPER_ADMIN_EMAILS", "").split(',') if e.strip()]
ALLOWED_CLIENT_EMAILS = [e.strip() for e in os.getenv("ALLOWED_CLIENT_EMAILS", "").split(',') if e.strip()]

# OAuth設定
oauth = OAuth()
if all([AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN]):
    oauth.register(
        name='auth0',
        client_id=AUTH0_CLIENT_ID,
        client_secret=AUTH0_CLIENT_SECRET,
        server_metadata_url=f'https://{AUTH0_DOMAIN}/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid profile email'},
    )