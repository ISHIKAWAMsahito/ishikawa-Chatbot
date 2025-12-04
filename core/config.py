import os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from authlib.integrations.starlette_client import OAuth
import logging

IS_PRODUCTION = os.getenv('RENDER', False)

if not IS_PRODUCTION:
    # ローカル開発環境の場合のみ .env ファイルを読み込む
    load_dotenv()
    logging.info("ローカル環境として .env ファイルを読み込みました。")
else:
    logging.info("本番環境として起動しました (Renderの環境変数を使用)。")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("環境変数 'GEMINI_API_KEY' が設定されていません。")

# Auth0設定
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
if not APP_SECRET_KEY:
    raise ValueError("環境変数 'APP_SECRET_KEY' が設定されていません。")

# Supabase設定
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# --- vvv ここから修正 vvv ---
# アプリケーション起動時に変数が None でないか強制的にチェックする
# これがインポートのタイミングの問題（分割バグ）か、
# Renderの環境変数設定ミス（タイプミス）かを切り分ける
if not SUPABASE_URL:
    logging.error("### 'SUPABASE_URL' が None です。インポート順の問題か、環境変数が設定されていません。 ###")
    raise ValueError("環境変数 'SUPABASE_URL' が設定されていません。")
    
if not SUPABASE_KEY:
    logging.error("### 'SUPABASE_SERVICE_KEY' が None です。インポート順の問題か、環境変数が設定されていません。 ###")
    raise ValueError("環境変数 'SUPABASE_SERVICE_KEY' が設定されていません。")
# --- ^^^ ここまで修正 ^^^ ---


# 定数
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
# デバッグ用：ログにキーの一部を出力して、どっちのキーを使っているか白黒つける
if GEMINI_API_KEY:
    print(f"DEBUG: Current API Key starts with: {GEMINI_API_KEY[:5]}...", flush=True)
else:
    print("DEBUG: GEMINI_API_KEY is empty!", flush=True)