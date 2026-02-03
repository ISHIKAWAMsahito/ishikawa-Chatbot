import os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from authlib.integrations.starlette_client import OAuth
import logging

# ----------------------------------------------------------------
# 1. 環境変数の読み込み設定
# ----------------------------------------------------------------
IS_PRODUCTION = os.getenv('RENDER', False)

if not IS_PRODUCTION:
    # ローカル開発環境: 指定されたフルパスから .env を読み込む
    # Windowsパスなので raw string (r"...") を使用
    env_path = r"C:\dev\ishikawa-Chatbot\ishikawa-Chatbot.env"
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logging.info(f"✅ ローカル環境: {env_path} から設定を読み込みました。")
    else:
        logging.warning(f"⚠️ 指定された .env ファイルが見つかりません: {env_path}")
        logging.info("デフォルトの load_dotenv() を試行します。")
        load_dotenv()
else:
    logging.info("🚀 本番環境として起動しました (Renderの環境変数を使用)。")

# ----------------------------------------------------------------
# 2. LangSmith (LangChain) 設定
# ----------------------------------------------------------------
# トレース有効化フラグ (文字列 "true" を bool に変換)
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ishikawa-chatbot-eval") # デフォルトプロジェクト名
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# 設定診断
if LANGCHAIN_TRACING_V2:
    if not LANGCHAIN_API_KEY:
        logging.warning("⚠️ LangSmithトレースは有効(TRACING_V2=true)ですが、APIキーが設定されていません。送信に失敗する可能性があります。")
    else:
        # セキュリティのためキーの一部のみ表示
        masked_ls_key = LANGCHAIN_API_KEY[:4] + "..."
        logging.info(f"🔎 LangSmith Tracing: ENABLED (Project: {LANGCHAIN_PROJECT}, Key: {masked_ls_key})")
else:
    logging.info("⚪ LangSmith Tracing: DISABLED")


# ----------------------------------------------------------------
# 3. APIキー & 認証設定
# ----------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # ローカルでパス指定ミスなどの可能性があるため、詳細なエラーを出す
    raise ValueError("環境変数 'GEMINI_API_KEY' が設定されていません。.envのパスや内容を確認してください。")

# Auth0設定
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
if not APP_SECRET_KEY:
    raise ValueError("環境変数 'APP_SECRET_KEY' が設定されていません。")

# ----------------------------------------------------------------
# 4. Supabase設定 (互換性対応版)
# ----------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# ★重要: main.py 等が古い変数名(SUPABASE_KEY)を参照していても動くようにエイリアスを設定
SUPABASE_KEY = SUPABASE_SERVICE_KEY 

# エラーチェック
if not SUPABASE_URL:
    raise ValueError("環境変数 'SUPABASE_URL' が設定されていません。")

if not SUPABASE_ANON_KEY:
    logging.warning("### 'SUPABASE_ANON_KEY' が設定されていません。学生画面の機能が一部制限される可能性があります。 ###")

if not SUPABASE_SERVICE_KEY:
    logging.error("### 'SUPABASE_SERVICE_KEY' が設定されていません。署名付きURLの発行ができません。 ###")
    raise ValueError("環境変数 'SUPABASE_SERVICE_KEY' (または SUPABASE_KEY) が設定されていません。")


# ----------------------------------------------------------------
# 5. その他定数
# ----------------------------------------------------------------
ACTIVE_COLLECTION_NAME = "student-knowledge-base"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JST = timezone(timedelta(hours=+9), 'JST')

SUPER_ADMIN_EMAILS_STR = os.getenv("SUPER_ADMIN_EMAILS", "")
SUPER_ADMIN_EMAILS = [email.strip() for email in SUPER_ADMIN_EMAILS_STR.split(',') if email.strip()]

ALLOWED_CLIENT_EMAILS_STR = os.getenv("ALLOWED_CLIENT_EMAILS", "")
ALLOWED_CLIENT_EMAILS = [email.strip() for email in ALLOWED_CLIENT_EMAILS_STR.split(',') if email.strip()]

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
else:
    logging.warning("Auth0の設定が不完全なため、管理者ページの認証機能は動作しません。")

# デバッグ用ログ
if GEMINI_API_KEY:
    # 下5桁を抽出するために負のインデックス [-5:] を使用
    masked_key = "..." + GEMINI_API_KEY[-5:]
    # flush=True を指定することで、Renderのログに即座に出力されるようにします
    # print(f"DEBUG: Current API Key ends with: {masked_key}", flush=True)
else:
    print("DEBUG: GEMINI_API_KEY is empty!", flush=True)