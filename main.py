# --------------------------------------------------------------------------
# 1. ライブラリのインポート
# --------------------------------------------------------------------------
import os
import json
import uvicorn
import traceback
import csv
import certifi
from datetime import datetime, timezone, timedelta
import uuid
import io
import logging
import asyncio
import re
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# --- サードパーティライブラリ ---
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, FileResponse, Response
from pydantic import BaseModel

from supabase import create_client, Client
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document as DocxDocument

from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth

from fastapi import Request
from fastapi.responses import FileResponse

from collections import defaultdict
from typing import Dict, List

# --- 初期設定 ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')
load_dotenv()

# --------------------------------------------------------------------------
# 2. 環境変数と基本設定
# --------------------------------------------------------------------------
# Gemini API設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("環境変数 'GEMINI_API_KEY' が設定されていません。")
genai.configure(api_key=GEMINI_API_KEY)

# Auth0設定 (これは元のコードのもので、新しいJWT認証とは別)
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "default-secret-key-for-development")

# Supabase設定
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# 定数
ACTIVE_COLLECTION_NAME = "student-knowledge-base"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JST = timezone(timedelta(hours=+9), 'JST')

SUPER_ADMIN_EMAILS_STR = os.getenv("SUPER_ADMIN_EMAILS", "")
SUPER_ADMIN_EMAILS = [email.strip() for email in SUPER_ADMIN_EMAILS_STR.split(',') if email.strip()]

ALLOWED_CLIENT_EMAILS_STR = os.getenv("ALLOWED_CLIENT_EMAILS", "")
ALLOWED_CLIENT_EMAILS = [email.strip() for email in ALLOWED_CLIENT_EMAILS_STR.split(',') if email.strip()]
# キーワードマッピング

# データベースから読み込むフォールバック情報を格納するグローバル変数
chat_histories: Dict[str, List[Dict[str, str]]] = defaultdict(list)
MAX_HISTORY_LENGTH = 20 
# --------------------------------------------------------------------------
# 3. 内部コンポーネントの定義
# --------------------------------------------------------------------------

def split_text(text: str, max_length: int = 1000, overlap: int = 200) -> List[str]:
    """テキストをチャンクに分割"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunks.append(text[start:end])
        start += max_length - overlap
    return chunks

def format_urls_as_links(text: str) -> str:
    """URLをHTMLリンクに変換"""
    url_pattern = r'(?<!\]\()(https?://[^\s\[\]()<>]+)(?!\))'
    text = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text)
    md_link_pattern = r'\[([^\]]+)\]\((https?://[^\s\)]+)\)'
    text = re.sub(md_link_pattern, r'<a href="\2" target="_blank">\1</a>', text)
    return text

async def safe_generate_content(model, prompt, stream=False, max_retries=3):
    """レート制限を考慮した安全なコンテンツ生成"""
    for attempt in range(max_retries):
        try:
            if stream:
                return await model.generate_content_async(
                    prompt,
                    stream=True,
                    generation_config=GenerationConfig(
                        max_output_tokens=1024,
                        temperature=0.70
                    )
                )
            else:
                return await model.generate_content_async(
                    prompt,
                    generation_config=GenerationConfig(
                        max_output_tokens=512,
                        temperature=0.3
                    )
                )
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = 15
                    if "retry in" in error_str:
                        try:
                            match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str)
                            if match:
                                wait_time = float(match.group(1)) + 2
                        except:
                            pass
                    logging.warning(f"API制限により{wait_time}秒待機中... (試行 {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise HTTPException(
                        status_code=429,
                        detail=f"APIクォータを超過しました。しばらく時間をおいてから再試行してください。"
                    )
            else:
                raise e
    raise HTTPException(status_code=500, detail="最大リトライ回数を超えました。")

class DocumentProcessor:
    """ドキュメント処理クラス"""
    def process(self, filename: str, content: bytes) -> List[str]:
        text = ""
        try:
            if filename.lower().endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            elif filename.lower().endswith(".docx"):
                doc = DocxDocument(io.BytesIO(content))
                for para in doc.paragraphs:
                    text += para.text + "\n"
            else:
                text = content.decode('utf-8', errors='ignore')
        except Exception as e:
            logging.error(f"ファイル処理エラー ({filename}): {e}")
            return []
        return split_text(text)

class WebScraper:
    """ウェブスクレイピングクラス"""
    def scrape(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10, verify=certifi.where())
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
                element.decompose()
            text = " ".join(t.strip() for t in soup.stripped_strings)
            return re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            logging.error(f"スクレイピングエラー ({url}): {e}")
            return ""

class FeedbackManager:
    """フィードバック管理クラス（簡素化版）"""
    def __init__(self):
        self.feedback_file = os.path.join(BASE_DIR, "feedback.json")

    def save_feedback(self, feedback_id: str, rating: str, comment: str = ""):
        """フィードバックを保存"""
        try:
            feedback_data = []
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
            feedback_entry = {
                "id": feedback_id,
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.now(JST).isoformat()
            }
            feedback_data.append(feedback_entry)
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, ensure_ascii=False, indent=2)
            logging.info(f"フィードバック保存完了: {feedback_id} - {rating}")
        except Exception as e:
            logging.error(f"フィードバック保存エラー: {e}")
            raise

    def get_feedback_stats(self) -> Dict[str, Any]:
        """フィードバック統計を取得"""
        try:
            if not os.path.exists(self.feedback_file):
                return {"total": 0, "resolved": 0, "not_resolved": 0, "rate": 0}
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            total = len(feedback_data)
            resolved = sum(1 for fb in feedback_data if fb['rating'] == 'resolved')
            not_resolved = total - resolved
            rate = (resolved / total * 100) if total > 0 else 0
            return {
                "total": total,
                "resolved": resolved,
                "not_resolved": not_resolved,
                "rate": round(rate, 1)
            }
        except Exception as e:
            logging.error(f"フィードバック統計取得エラー: {e}")
            return {"total": 0, "resolved": 0, "not_resolved": 0, "rate": 0}

class SupabaseClientManager:
    """Supabaseクライアント管理クラス"""
    def search_documents_by_vector(self, collection_name: str, embedding: List[float], match_count: int) -> List[dict]:
        """カテゴリで絞り込まずにベクトル検索を行う"""
        params = {
            "p_collection_name": collection_name,
            "p_query_embedding": embedding,
            "p_match_count": match_count
        }
        # 作成した新しいRPC関数 'match_documents_by_vector' を呼び出す
        result = self.client.rpc("match_documents_by_vector", params).execute()
        return result.data or []
    def search_fallback_qa(self, embedding: List[float], match_count: int) -> List[dict]:
        """Q&Aフォールバックをベクトル検索する"""
        params = {
            "p_query_embedding": embedding,
            "p_match_count": match_count
        }
        # ステップ1で作成した 'match_fallback_qa' を呼び出す
        result = self.client.rpc("match_fallback_qa", params).execute()
        return result.data or []
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def get_db_type(self) -> str:
        return "supabase"

    def insert_document(self, content: str, embedding: List[float], metadata: dict):
        self.client.table("documents").insert({
            "content": content,
            "embedding": embedding,
            "metadata": metadata
        }).execute()

    def search_documents(self, collection_name: str, category: str, embedding: List[float], match_count: int) -> List[dict]:
        params = {
            "p_collection_name": collection_name,
            "p_category": category,
            "p_query_embedding": embedding,
            "p_match_count": match_count
        }
        result = self.client.rpc("match_documents", params).execute()
        return result.data or []

    def get_documents_by_collection(self, collection_name: str) -> List[dict]:
        result = self.client.table("documents").select("id, metadata").eq("metadata->>collection_name", collection_name).execute()
        return result.data or []

    def count_chunks_in_collection(self, collection_name: str) -> int:
        result = self.client.table("documents").select("id", count='exact').eq("metadata->>collection_name", collection_name).execute()
        return result.count or 0

    def get_distinct_categories(self, collection_name: str) -> List[str]:
        try:
            result = self.client.rpc("get_distinct_categories", {"p_collection_name": collection_name}).execute()
            categories = [item['category'] for item in (result.data or []) if item.get('category')]
            return categories if categories else ["その他"]
        except Exception as e:
            logging.error(f"RPC 'get_distinct_categories' の呼び出しエラー: {e}")
            return ["その他"]

class SettingsManager:
    """設定管理クラス"""
    def __init__(self):
        self.settings = {
            "model": "gemini-2.5-flash",
            "collection": ACTIVE_COLLECTION_NAME,
            "embedding_model": "text-embedding-004",
            "top_k": 5
        }
        self.websocket_connections: List[WebSocket] = []
        self.settings_file = os.path.join(BASE_DIR, "shared_settings.json")
        self.load_settings()

    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.settings.update(json.load(f))
        except Exception as e:
            logging.error(f"設定ファイルの読み込みエラー: {e}")

    def save_settings(self):
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"設定ファイルの保存エラー: {e}")

    async def update_settings(self, new_settings: Dict[str, Any]):
        self.settings.update(new_settings)
        self.save_settings()
        await self.broadcast_settings()

    async def add_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.websocket_connections.append(websocket)

    def remove_websocket(self, websocket: WebSocket):
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)

    async def broadcast_settings(self):
        message = {"type": "settings_update", "data": self.settings}
        disconnected = []
        for conn in self.websocket_connections:
            try:
                await conn.send_json(message)
            except:
                disconnected.append(conn)
        for conn in disconnected:
            self.remove_websocket(conn)


# --------------------------------------------------------------------------
# 4. FastAPIアプリケーションのセットアップ
# --------------------------------------------------------------------------
db_client: Optional[SupabaseClientManager] = None
settings_manager: Optional[SettingsManager] = None


# 8. lifespan関数を更新
@asynccontextmanager
async def lifespan(app: FastAPI):
    """認証システムを含むアプリケーションのライフサイクル管理"""
    # 'g_category_fallbacks' を global 宣言から削除
    global db_client, settings_manager
    logging.info("--- アプリケーション起動処理開始(認証システム対応) ---")

    settings_manager = SettingsManager()

    if SUPABASE_URL and SUPABASE_KEY:
        try:
            db_client = SupabaseClientManager(url=SUPABASE_URL, key=SUPABASE_KEY)
            logging.info("Supabaseクライアントの初期化完了。")



        except Exception as e:
            logging.error(f"Supabase初期化エラー: {e}")
    else:
        logging.warning("Supabaseの環境変数が設定されていません。")

    yield

    logging.info("--- アプリケーション終了処理(認証システム対応) ---")

app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=APP_SECRET_KEY)
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)

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

# --- グローバルインスタンス ---
feedback_manager = FeedbackManager()
document_processor = DocumentProcessor()
web_scraper = WebScraper()

# --- データモデル定義 ---
class ChatQuery(BaseModel):
    query: str
    model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-004"
    top_k: int = 5
    collection: str = ACTIVE_COLLECTION_NAME

class ClientChatQuery(BaseModel):
    query: str

class ScrapeRequest(BaseModel):
    url: str
    category: str = "その他"
    embedding_model: str = "text-embedding-004"
    collection_name: str = ACTIVE_COLLECTION_NAME


class FeedbackRequest(BaseModel):
    feedback_id: str
    rating: str
    comment: str = ""

class AIResponseFeedbackRequest(BaseModel):
    user_question: str
    ai_response: str
    rating: str  # 'good' or 'bad'

class Settings(BaseModel):
    model: Optional[str] = None
    collection: Optional[str] = None
    embedding_model: Optional[str] = None
    top_k: Optional[int] = None

# WebSocket接続管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                self.disconnect(connection)
manager = ConnectionManager()

# --------------------------------------------------------------------------
# 5. APIエンドポイント定義
# --------------------------------------------------------------------------
@app.post("/feedback/ai-response")
async def save_ai_response_feedback(feedback: AIResponseFeedbackRequest):
    try:
        # feedback.comment にAI回答 or ユーザーコメントを格納
        # created_atはPython側でISOフォーマットのタイムスタンプを格納
        # ratingは good/bad など
        db_client.client.table("anonymous_comments").insert({
            "comment": f"Q: {feedback.user_question}\nA: {feedback.ai_response}",  # 質問・回答をまとめてテキスト化
            "created_at": datetime.now(JST).isoformat(),
            "rating": feedback.rating
        }).execute()
        return {"message": "AI回答へのフィードバックを保存しました"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# main.py のどこか（APIエンドポイントが定義されているエリア）に追加

@app.get("/config")
def get_config():
    """
    フロントエンドが必要とする公開可能な設定（環境変数）を返す
    """
    return {
        "supabase_url": os.getenv("SUPABASE_URL"),
        "supabase_anon_key": os.getenv("SUPABASE_ANON_KEY")
    }
    


    
# --- 認証関数 (Auth0用) ---
# --- 管理者用認証 ---
def require_auth(request: Request):
    """管理者用認証 (SUPER_ADMIN_EMAILS のみ許可)"""
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=307, headers={'Location': '/login'})
    
    # 念のため小文字に変換して比較
    user_email = user.get('email', '').lower()
    super_admin_emails_lower = [email.lower() for email in SUPER_ADMIN_EMAILS]

    if user_email in super_admin_emails_lower:
        return user
    else:
        # スーパー管理者リストに含まれていない場合は、管理者ページへのアクセスを拒否
        raise HTTPException(status_code=403, detail="管理者ページへのアクセス権がありません。")

# --- 学生用認証 ---
def require_auth_client(request: Request):
    """クライアント用認証 (ALLOWED_CLIENT_EMAILS または SUPER_ADMIN_EMAILS を許可)"""
    user = request.session.get('user')

    # 1. 最初に、ログインしているかどうかを確認します。
    if not user:
        raise HTTPException(status_code=307, headers={'Location': '/login'})

    # 2. ユーザー情報があることが確定してから、メールアドレスを取得します。
    user_email = user.get('email', '').lower()

    # 3. 比較対象の許可リストを両方とも小文字に変換しておきます。
    allowed_emails_lower = [email.lower() for email in ALLOWED_CLIENT_EMAILS]
    super_admin_emails_lower = [email.lower() for email in SUPER_ADMIN_EMAILS]

    # --- デバッグ用のprint文（安全な場所に移動） ---
    # print("--- クライアント認証チェック ---")
    # print(f"ログイン試行中のメアド (小文字化後): '[{user_email}]'")
    # print(f"クライアント許可リスト: {allowed_emails_lower}")
    # print(f"管理者許可リスト: {super_admin_emails_lower}")
    # print(f"クライアントリストに含まれているか？: {user_email in allowed_emails_lower}")
    # print(f"管理者リストに含まれているか？: {user_email in super_admin_emails_lower}")
    # print("--------------------")
    # -----------------------------------------------------------

    # 4. 認証チェックを実行します。
    # (クライアント許可リスト、または管理者許可リストのどちらかに含まれていればOK)
    if (user_email in allowed_emails_lower or
        user_email in super_admin_emails_lower):
        return user
    else:
        raise HTTPException(status_code=403, detail="このサービスへのアクセスは許可されていません。")

# --- 認証とHTML提供 (Auth0用) ---
# main.pyの修正箇所（ルート定義部分のみ）
# 既存の重複したルート定義を削除し、以下に置き換えてください

# --- 認証とHTML提供 (Auth0用) ---
@app.get('/login')
async def login_auth0(request: Request):
    if 'auth0' not in oauth._clients:
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    return await oauth.auth0.authorize_redirect(request, request.url_for('auth'))

@app.get('/auth')
async def auth(request: Request):
    """
    Auth0からのコールバックを処理し、ユーザー情報に基づいて
    適切なページ（/admin または /）にリダイレクトする。
    """
    if 'auth0' not in oauth._clients:
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    
    try:
        token = await oauth.auth0.authorize_access_token(request)
    except Exception as e:
        logging.error(f"Auth0 access token error: {e}")
        return RedirectResponse(url='/login')

    if userinfo := token.get('userinfo'):
        # ユーザー情報をセッションに保存
        request.session['user'] = dict(userinfo)
        user_email = userinfo.get('email', '').lower() # 小文字に変換
        
        # 許可リストを小文字に
        super_admin_emails_lower = [email.lower() for email in SUPER_ADMIN_EMAILS]
        allowed_emails_lower = [email.lower() for email in ALLOWED_CLIENT_EMAILS]

        # --- 権限に基づくリダイレクト判定 ---
        
        # 1. まず、管理者(SUPER_ADMIN)か？ (最優先)
        #    管理者は /admin にリダイレクト
        if user_email in super_admin_emails_lower:
            return RedirectResponse(url='/admin')

        # 2. 次に、クライアント(学生など)か？
        #    クライアントは / (学生用画面) にリダイレクト
        elif user_email in allowed_emails_lower:
            return RedirectResponse(url='/')
        
        # 3. 上記のいずれにも該当しない場合は、許可されていないユーザー
        #    ログアウトさせてアクセスを拒否
        else:
            logging.warning(f"Unauthorized login attempt by: {user_email}")
            return RedirectResponse(url='/logout')
            
    # Auth0からuserinfoが取得できなかった場合
    logging.error("Failed to get userinfo from Auth0.")
    return RedirectResponse(url='/login')

@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    if not all([AUTH0_DOMAIN, AUTH0_CLIENT_ID]):
        return RedirectResponse(url='/')
    return RedirectResponse(f"https://{AUTH0_DOMAIN}/v2/logout?returnTo={request.url_for('serve_client')}&client_id={AUTH0_CLIENT_ID}")

# クライアント用ページ（学生用）
@app.get("/", response_class=FileResponse)
async def serve_client(request: Request, user: dict = Depends(require_auth_client)):
    return FileResponse(os.path.join(BASE_DIR, "client.html"))

# 管理者用メインページ
@app.get("/admin", response_class=FileResponse)
async def serve_admin(request: Request, user: dict = Depends(require_auth)):
    return FileResponse(os.path.join(BASE_DIR, "admin.html"))

# ★★★ 重要: DB管理ページ（管理者のみアクセス可能）★★★
# 重複を削除し、この1つだけ残します
@app.get("/DB.html", response_class=FileResponse)
async def serve_db_page(request: Request, user: dict = Depends(require_auth)):
    """DB管理ページ(DB.html)を提供するためのエンドポイント"""
    db_path = os.path.join(BASE_DIR, "DB.html")
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail="DB.html not found")
    return FileResponse(db_path)

# CSS提供
@app.get("/style.css", response_class=FileResponse)
async def serve_css():
    """統一されたCSSファイル(style.css)を提供するためのエンドポイント"""
    return FileResponse(os.path.join(BASE_DIR, "style.css"))

# フィードバック統計ページ
@app.get("/feedback-stats", response_class=FileResponse)
async def serve_feedback_stats(request: Request, user: dict = Depends(require_auth)):
    return FileResponse(os.path.join(BASE_DIR, "feedback_stats.html"))

# Faviconエラー回避
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

# --- ステータス確認 ---
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "database": db_client.get_db_type() if db_client else "uninitialized"
    }

@app.get("/gemini/status")
async def gemini_status():
    if not GEMINI_API_KEY:
        return {"connected": False, "detail": "API key not configured"}
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return {"connected": True, "models": models}
    except Exception as e:
        return {"connected": False, "detail": str(e)}

@app.api_route("/healthz", methods=["GET", "HEAD"])
def health_check_k8s():
    return {"status": "ok"}

# --- コレクション管理API ---
@app.get("/collections")
async def get_collections():
    return [{"name": ACTIVE_COLLECTION_NAME, "count": db_client.count_chunks_in_collection(ACTIVE_COLLECTION_NAME) if db_client else 0}]

@app.post("/collections")
async def create_collection(request: dict):
    return {"message": f"コレクション「{ACTIVE_COLLECTION_NAME}」は既に存在しています"}

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    if collection_name == ACTIVE_COLLECTION_NAME:
        raise HTTPException(status_code=400, detail="このコレクションは削除できません")
    return {"message": "コレクションが見つかりません"}

# --- ナレッジ管理API ---
# --- ナレッジ管理API ---
@app.get("/collections/{collection_name}/documents")
async def get_documents(collection_name: str):
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    return {
        "documents": db_client.get_documents_by_collection(collection_name),
        "count": db_client.count_chunks_in_collection(collection_name)
    }

# --- ドキュメント管理API（新規追加） ---
@app.get("/api/documents/all")
async def get_all_documents(
    user: dict = Depends(require_auth),
    page: int = Query(1, ge=1),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None),
    category: Optional[str] = Query(None)
):
    """
    全ドキュメントをページネーション、検索、カテゴリフィルタ対応で取得（管理者のみ）
    """
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        # --- 1. ベースとなるクエリを構築 ---
        # (SupabaseのPostgRESTビルダーを使って、動的にクエリを組み立てます)
        query = db_client.client.table("documents")
        count_query = db_client.client.table("documents")

        # --- 2. フィルタ条件の適用 ---
        if category:
            query = query.eq("metadata->>category", category)
            count_query = count_query.eq("metadata->>category", category)

        if search:
            # content, category, source のいずれかに一致
            search_term = f"%{search}%"
            query = query.or_(
                f"content.ilike.{search_term}",
                f"metadata->>category.ilike.{search_term}",
                f"metadata->>source.ilike.{search_term}"
            )
            count_query = count_query.or_(
                f"content.ilike.{search_term}",
                f"metadata->>category.ilike.{search_term}",
                f"metadata->>source.ilike.{search_term}"
            )

        # --- 3. 総件数の取得 (フィルタ適用後) ---
        # .select("id", count='exact') で、実際のデータを取得せず件数だけを取得
        count_response = count_query.select("id", count='exact').execute()
        total_records = count_response.count or 0

        # --- 4. ページ指定されたデータの取得 ---
        # Supabase (PostgREST) の range は limit/offset 方式
        offset = (page - 1) * limit
        # .range(from, to) ※ to は inclusive
        data_response = query.select("*").order("id", desc=True).range(offset, offset + limit - 1).execute()

        return {
            "documents": data_response.data or [],
            "total": total_records,
            "page": page,
            "limit": limit
        }

    except Exception as e:
        logging.error(f"ドキュメント一覧(ページネーション)取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{doc_id}")
async def get_document_by_id(doc_id: int, user: dict = Depends(require_auth)):
    """特定のドキュメントを取得（管理者のみ）"""
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = db_client.client.table("documents").select("*").eq("id", doc_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ドキュメント取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/documents/{doc_id}")
async def update_document(doc_id: int, request: Dict[str, Any], user: dict = Depends(require_auth)):
    """ドキュメントを更新（管理者のみ）。content更新時にベクトルも再生成する。"""
    if not db_client or not settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    try:
        update_data = {}

        # 1. contentが更新される場合は、ベクトルも再生成する
        if "content" in request:
            new_content = request["content"]
            update_data["content"] = new_content
            
            # 設定マネージャーから現在のエンベディングモデルを取得
            embedding_model = settings_manager.settings.get("embedding_model", "text-embedding-004")
            
            logging.info(f"ドキュメント {doc_id} のコンテンツが変更されたため、ベクトルを再生成します...")
            try:
                # 新しいコンテンツでベクトルを生成
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=new_content
                )
                update_data["embedding"] = embedding_response["embedding"]
                logging.info(f"ドキュメント {doc_id} のベクトル再生成が完了しました。")
            except Exception as e:
                # APIのレート制限などを考慮
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning("ベクトル再生成でAPI制限に達しました。30秒待機します。")
                    await asyncio.sleep(30)
                    embedding_response = genai.embed_content(
                        model=embedding_model,
                        content=new_content
                    )
                    update_data["embedding"] = embedding_response["embedding"]
                else:
                    logging.error(f"ベクトル再生成エラー: {e}")
                    raise HTTPException(status_code=500, detail=f"ベクトル再生成中にエラーが発生しました: {e}")

        # 2. メタデータは独立して更新可能
        if "metadata" in request:
            update_data["metadata"] = request["metadata"]

        if not update_data:
            raise HTTPException(status_code=400, detail="更新するデータがありません")
        
        result = db_client.client.table("documents").update(update_data).eq("id", doc_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
        
        logging.info(f"ドキュメント {doc_id} を更新しました（管理者: {user.get('email')}）")
        return {"message": "ドキュメントを更新しました", "document": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ドキュメント更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int, user: dict = Depends(require_auth)):
    """ドキュメントを削除（管理者のみ）"""
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = db_client.client.table("documents").delete().eq("id", doc_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="ドキュメントが見つかりません")
        
        logging.info(f"ドキュメント {doc_id} を削除しました（管理者: {user.get('email')}）")
        return {"message": "ドキュメントを削除しました"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"ドキュメント削除エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(
     user: dict = Depends(require_auth),
    file: UploadFile = File(...),
    embedding_model: str = Form("text-embedding-004"),
    collection_name: str = Form(ACTIVE_COLLECTION_NAME)
):
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        category = file.filename.split('_')[0] if '_' in file.filename else "その他"
        content = await file.read()
        chunks = document_processor.process(file.filename, content)
        for chunk in chunks:
            metadata = {
                "source": file.filename,
                "collection_name": collection_name,
                "category": category
            }
            try:
                embedding_response = genai.embed_content(model=embedding_model, content=chunk)
                embedding = embedding_response["embedding"]
                db_client.insert_document(chunk, embedding, metadata)
                await asyncio.sleep(1)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning("埋め込み生成でAPI制限に達しました。30秒待機します。")
                    await asyncio.sleep(30)
                    embedding_response = genai.embed_content(model=embedding_model, content=chunk)
                    embedding = embedding_response["embedding"]
                    db_client.insert_document(chunk, embedding, metadata)
                else:
                    raise
        return {"chunks": len(chunks)}
    except Exception as e:
        logging.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrape")
async def scrape_website(req: ScrapeRequest):
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        content = web_scraper.scrape(req.url)
        if not content:
            raise HTTPException(status_code=400, detail="コンテンツを取得できませんでした。")
        chunks = split_text(content)
        for chunk in chunks:
            metadata = {
                "source": req.url,
                "collection_name": req.collection_name,
                "category": req.category
            }
            try:
                embedding_response = genai.embed_content(model=req.embedding_model, content=chunk)
                embedding = embedding_response["embedding"]
                db_client.insert_document(chunk, embedding, metadata)
                await asyncio.sleep(1)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning("スクレイピングでAPI制限に達しました。30秒待機します。")
                    await asyncio.sleep(30)
                    embedding_response = genai.embed_content(model=req.embedding_model, content=chunk)
                    embedding = embedding_response["embedding"]
                    db_client.insert_document(chunk, embedding, metadata)
                else:
                    raise
        return {"chunks": len(chunks)}
    except Exception as e:
        logging.error(f"Scrape error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
 # --- チャットAPI（RAG対応） ---

# 686行目から734行目までの不完全なenhanced_chat_logic関数を削除

# その後、738行目付近に以下の関数を追加(グローバルスコープ)
def get_or_create_session_id(request: Request) -> str:
    """セッションIDを取得または新規作成"""
    session_id = request.session.get('chat_session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        request.session['chat_session_id'] = session_id
    return session_id

def add_to_history(session_id: str, role: str, content: str):
    """チャット履歴に追加"""
    # ---------- ここを一時的に無効化しています ----------
    # 会話の記録を完全に停止したい（コメントアウトで一時無効化）
    # 元に戻す際は、下のコードをコメント解除してください。
    #
    # chat_histories[session_id].append({
    #     "role": role,
    #     "content": content
    # })
    # if len(chat_histories[session_id]) > MAX_HISTORY_LENGTH:
    #     chat_histories[session_id] = chat_histories[session_id][-MAX_HISTORY_LENGTH:]
    return
# ------------------------------------------------------

def get_history(session_id: str) -> List[Dict[str, str]]:
    """チャット履歴を取得"""
    return chat_histories.get(session_id, [])

def clear_history(session_id: str):
    """チャット履歴をクリア"""
    if session_id in chat_histories:
        del chat_histories[session_id]

# 次に、enhanced_chat_logic関数を正しく定義
# main.py の916行目あたりから

# [main.py] 916行目から
async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    user_input = chat_req.query.strip()
    feedback_id = str(uuid.uuid4())
    
    # セッションIDを取得
    session_id = get_or_create_session_id(request)
    
    yield f"data: {json.dumps({'feedback_id': feedback_id})}\n\n"
    
    try:
        if not all([db_client, GEMINI_API_KEY]):
            error_msg = "システムが利用できません。管理者にお問い合わせください。"
            yield f"data: {json.dumps({'content': error_msg})}\n\n"
            return

        context = ""
        has_specific_info = False
        MIN_SIMILARITY_THRESHOLD = 0.7 # 類似度のしきい値
        search_results = [] # ★★★ ここで初期化 ★★★

        # --- メインの検索ロジック：ベクトル検索を優先 ---
        try:
            query_embedding_response = genai.embed_content(model=chat_req.embedding_model, content=user_input)
            query_embedding = query_embedding_response["embedding"]
            
            # ★★★ 修正点: ここにベクトル検索の実行処理を追加 ★★★
            if db_client:
                search_results = db_client.search_documents_by_vector(
                    collection_name=chat_req.collection,
                    embedding=query_embedding,
                    match_count=chat_req.top_k
                )
            # ★★★ 修正ここまで ★★★

            # 類似度がしきい値以上の結果「のみ」を抽出する
            relevant_docs = [
                doc for doc in search_results 
                if doc.get('content') and doc.get('similarity', 0) >= MIN_SIMILARITY_THRESHOLD
            ]

            # 関連ドキュメントが1件以上存在する場合のみ、RAGを実行する
# 関連ドキュメントが1件以上存在する場合のみ、RAGを実行する
            if relevant_docs:
                # 関連ドキュメントが存在する場合
                context_parts = []
                for doc in relevant_docs:
                    # Supabaseの 'documents' テーブルの metadata カラムから 'source' を取得
                    source_info = doc.get('metadata', {}).get('source', '不明なソース')
                    
                    # ★★★ AIが参照できるように「出典」と「内容」をセットにする ★★★
                    context_parts.append(f"<document source=\"{source_info}\">\n{doc['content']}\n</document>")
                
                # 各ドキュメントを区切り線で分割して context を構築
                context = "\n\n---\n\n".join(context_parts)
                has_specific_info = True

        except Exception as e:
            logging.error(f"データベース検索エラー: {e}")

            prompt = f"""あなたは「札幌学院大学」の学生サポート専門、RAG（Retrieval-Augmented Generation）AIアシスタントです。
あなたのタスクは、提供された <context> のみに基づいて、ユーザーの <query> に回答することです。

# 厳守ルール (System Rules)
1.  **絶対的厳守:** 回答は、<context> タグ内に提供された情報**のみ**に基づいてください。あなたの一般知識、トレーニングデータ、<context> 外の情報は**一切使用禁止**です。
2.  **出典の明記 (Citation):**
    * <context> 内の情報は `<document source="...">` タグで提供されます。
    * 回答で引用した情報（文章、事実）の**直後に**、必ずその出典を `[出典: ...]` の形式で付記してください。
    * 複数文をまとめて引用する場合は、段落末尾にまとめて `[出典: ...]` を付記しても構いません。
    * 例: 「履修登録はWebから行います。[出典: G-PLUSガイド.pdf]」
3.  **推測の禁止:** <context> に書かれていない手続きの期限、担当部署、必要書類などを、あなたの知識で補完してはいけません。書かれている場合のみ、そのまま引用してください。

# 回答手順 (Thinking Process)
AIは以下のステップに従って回答を生成してください。

[ステップ1: クエリ分析]  
ユーザーの <query> が何を要求しているかを正確に分析します。  
(例: 「2年生 経済学部 休学 留学 4年で卒業 履修」)

[ステップ2: コンテキスト検索]  
<context> 内の全ドキュメントをスキャンし、[ステップ1]のキーワードに関連する情報ブロックをすべて特定します。

[ステップ3: 関連性判定と統合]  
特定した情報を統合し、<query> 全体に答えられるか厳密に判断します。  
- 「十分」: 質問に直接答える情報が揃っている。  
- 「不十分」: 質問の一部にしか答えられない、または関連する情報が全くない。  

[ステップ4: 回答生成]  
* **(判定が「十分」の場合) → ケースA**  
    1.  `「データベースの情報に基づき、ご質問にお答えします。」` という書き出しで始めます。  
    2.  [ステップ2]で特定した情報**のみ**を用いて、質問に対する具体的で分かりやすい回答を構成します。  
    3.  ルール(2)に従い、すべての情報に出典 `[出典: ...]` を付記します。  

* **(判定が「不十分」の場合) → ケースB**  
    1.  `「ご質問に直接お答えできる情報は見つかりませんでしたが、関連する情報として以下をご確認ください。」` と書き出します。  
    2.  [ステップ2]で実際に見つかった部分的な関連情報のみを提示します。（情報が見つからなかった場合は省略）  
    3.  出典 `[出典: ...]` を必ず付記します。  
    4.  部署名や窓口が <context> に記載されていない場合は「大学の公式窓口にご確認ください」と案内します。  

# 出力フォーマット
- 学生にとって分かりやすい、親切で丁寧な「です・ます調」を使用します。  
- 見出しや箇条書きを活用し、要点を整理して提示します。  
- <context> 内にURL（http...）が記載されていた場合は、回答末尾に「参考URL:」として箇条書きで記載します。  

---
[ここからがデータです]

<context>
{context}
</context>

<query>
{user_input}
</query>

---
[あなたの回答]  
回答:

"""
            model = genai.GenerativeModel(chat_req.model)
            
            temp_full_response = ""
            try:
                # ======================================================
                # ★★★ ここが修正点です ★★★
                # stream の初期化(元1063行目)を try ブロックの内側に移動
                stream = await safe_generate_content(model, prompt, stream=True)
            
                async for chunk in stream:
                    if chunk.text:
                        temp_full_response += chunk.text
            
            except StopAsyncIteration:
                # これで、streamの初期化時に空のストリームが返されても
                # 正常にキャッチできるようになります。
                logging.warning("APIから空のストリームが返されました。セーフティ設定によるブロックの可能性があります。")
                temp_full_response = "申し訳ありませんが、そのご質問にはお答えできません。質問の内容を変えて、もう一度お試しください。"
            # ======================================================

            if not temp_full_response.strip():
                 temp_full_response = "回答を生成できませんでした。もう一度お試しください。"

# ↓↓↓ main.pyの 1060行目あたりから置き換える ↓↓↓

            full_response = format_urls_as_links(temp_full_response)
            
            # add_to_history は一時無効化済み（no-op）なので呼び出しても記録されません
            add_to_history(session_id, "user", user_input)
            add_to_history(session_id, "assistant", temp_full_response)
            
            yield f"data: {json.dumps({'content': full_response})}\n\n"
        
        # --- ここからが「キーワードマップ廃止」の置き換え ---
        else:
            # --- フォールバック処理 (Stage 2: Q&Aベクトル検索) ---
            # メインのRAG検索(Stage 1)で、類似度65%以上の情報が見つからなかった
            logging.info(f"Stage 1 RAG 失敗。Stage 2 (Q&Aベクトル検索) を実行します。")
            
            try:
                # Stage 1 で使用した query_embedding を再利用 (943行目で生成済み)
                fallback_results = db_client.search_fallback_qa(
                    embedding=query_embedding, 
                    match_count=1  # 最も近いQ&Aを1つだけ取得
                )
                
                if fallback_results:
                    # Q&Aベクトル検索で最も近いものが見つかった
                    best_match = fallback_results[0]
                    
                    # Q&Aに対する類似度の「足切りライン」
                    FALLBACK_SIMILARITY_THRESHOLD = 0.65
                    
                    if best_match.get('similarity', 0) >= FALLBACK_SIMILARITY_THRESHOLD:
                        logging.info(f"Stage 2 RAG 成功。類似Q&Aを回答します (Similarity: {best_match['similarity']:.2f})")
                        
                        # Q&Aが見つかった場合の定型文
                        fallback_response = f"""データベースに直接の情報は見つかりませんでしたが、関連する「よくあるご質問」がありましたのでご案内します。

---
{best_match['content']}
"""
                        full_response = format_urls_as_links(fallback_response)
                    else:
                        # Q&A検索もしたが、類似度が低すぎた
                        logging.info(f"Stage 2 RAG 失敗。類似するQ&Aが見つかりませんでした (Best Similarity: {best_match.get('similarity', 0):.2f})")
                        fallback_response = "申し訳ありませんが、ご質問に関連する情報がデータベース（Q&Aを含む）に見つかりませんでした。大学公式サイトをご確認いただくか、学生支援課までお問い合わせください。"
                        full_response = format_urls_as_links(fallback_response)

                else:
                    # Q&A検索で何もヒットしなかった（DBが空など）
                    logging.info(f"Stage 2 RAG 失敗。Q&Aデータベースが空か、検索エラーです。")
                    fallback_response = "申し訳ありませんが、ご質問に関連する情報が見つかりませんでした。大学公式サイトをご確認いただくか、学生支援課までお問い合わせください。"
                    full_response = format_urls_as_links(fallback_response)

            except Exception as e_fallback:
                logging.error(f"Stage 2 (Q&A検索) でエラーが発生: {e_fallback}")
                fallback_response = "申し訳ありません。現在、関連情報の検索中にエラーが発生しました。時間をおいて再度お試しください。"
                full_response = format_urls_as_links(fallback_response)

            
            # add_to_history は no-op のため記録されません
            add_to_history(session_id, "user", user_input)
            add_to_history(session_id, "assistant", fallback_response) # 生のテキストを記録
            
            yield f"data: {json.dumps({'content': full_response})}\n\n"
        # --- 置き換えはここまで ---


        yield f"data: {json.dumps({'show_feedback': True, 'feedback_id': feedback_id})}\n\n"
    
    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        logging.error(f"チャットロジックエラー: {e}\n{traceback.format_exc()}")
        yield f"data: {json.dumps({'content': error_message})}\n\n"
# --- ここまでが enhanced_chat_logic 関数 ---

# --- 置き換えはここまで (enhanced_chat_logic 関数の末尾) ---
# --- ここまでが enhanced_chat_logic 関数 ---

# 履歴管理用エンドポイント
@app.get("/chat/history")
async def get_chat_history(request: Request, user: dict = Depends(require_auth_client)):
    """現在のセッションのチャット履歴を取得"""
    session_id = get_or_create_session_id(request)
    history = get_history(session_id)
    return {"history": history}

@app.delete("/chat/history")
async def delete_chat_history(request: Request, user: dict = Depends(require_auth_client)):
    """現在のセッションのチャット履歴を削除"""
    session_id = get_or_create_session_id(request)
    clear_history(session_id)
    return {"message": "履歴をクリアしました"}

@app.post("/chat")
async def chat_endpoint(request: Request, query: ChatQuery):
    return StreamingResponse(enhanced_chat_logic(request, query), media_type="text/event-stream")

# 7. 既存のチャットエンドポイントを認証必須に変更
@app.post("/chat_for_client")
async def chat_for_client_auth(request: Request, query: ClientChatQuery, user: dict = Depends(require_auth_client)):
    """認証されたクライアント用チャットエンドポイント"""
    if not settings_manager:
        raise HTTPException(503, "Settings manager not initialized")

    logging.info(f"Chat request from user: {user.get('email', 'N/A')}")

    chat_query = ChatQuery(
        query=query.query,
        model=settings_manager.settings.get("model", "gemini-2.5-flash"),
        embedding_model=settings_manager.settings.get("embedding_model", "text-embedding-004"),
        top_k=settings_manager.settings.get("top_k", 5),
        collection=settings_manager.settings.get("collection", ACTIVE_COLLECTION_NAME)
    )
    return StreamingResponse(enhanced_chat_logic(request, chat_query), media_type="text/event-stream")


# --- フィードバックAPI ---
@app.post("/feedback")
async def save_feedback(feedback: FeedbackRequest):
    try:
        feedback_manager.save_feedback(feedback.feedback_id, feedback.rating, feedback.comment)
        return {"message": "フィードバックを保存しました"}
    except Exception as e:
        logging.error(f"フィードバック保存エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Q&A (category_fallbacks) 管理API ---
# main.py のAPIエンドポイント定義エリア (例: 1150行目あたり) に追加してください

@app.get("/api/fallbacks")
async def get_all_fallbacks(user: dict = Depends(require_auth)):
    """Q&A(フォールバック)をすべて取得"""
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = db_client.client.table("category_fallbacks").select("*").order("id", desc=True).execute()
        return {"fallbacks": result.data or []}
    except Exception as e:
        logging.error(f"Q&A一覧取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# [main.py] 1162行目あたり

# [main.py] 1162行目あたり

@app.post("/api/fallbacks")
async def create_fallback(request: Dict[str, Any], user: dict = Depends(require_auth)):
    """新しいQ&Aを作成（保存時に自動でベクトル化）"""
    # 依存関係に settings_manager を追加
    if not db_client or not settings_manager: 
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    
    try:
        new_qa_text = request.get("static_response", "")
        category_name = request.get("category_name") 

        if not new_qa_text:
            raise HTTPException(status_code=400, detail="static_response (Q&Aテキスト) は必須です")
        
        if not category_name:
            raise HTTPException(status_code=400, detail="category_name は必須です")

        # --- ここから自動ベクトル化ロジック ---
        embedding = None
        try:
            # 設定マネージャーからエンベディングモデルを取得
            embedding_model = settings_manager.settings.get("embedding_model", "text-embedding-004")
            logging.info(f"新規Q&Aのベクトルを生成します...")
            
            # ベクトルを生成
            embedding_response = genai.embed_content(
                model=embedding_model,
                content=new_qa_text
            )
            embedding = embedding_response["embedding"]
            logging.info(f"新規Q&Aのベクトル生成が完了しました。")
        
        except Exception as e:
            # レート制限などで失敗しても、テキストの保存は続行する (embedding = None のまま)
            logging.error(f"新規Q&Aのベクトル生成エラー: {e}")
            logging.warning(f"ベクトル化に失敗しましたが、テキストは保存します。")
        # --- ここまで ---

        # embedding を挿入データに含める
        insert_data = {
            "static_response": new_qa_text,
            "category_name": category_name,
            "url_to_summarize": request.get("url_to_summarize"), # (古いカラムが残っている場合)
            "embedding": embedding  # (None またはベクトル)
        }
        
        result = db_client.client.table("category_fallbacks").insert(insert_data).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Q&Aの作成に失敗しました")

        logging.info(f"新規Q&A {result.data[0]['id']} を作成しました（管理者: {user.get('email')}）")
        
        # メッセージを変更
        message = "新しいQ&Aを保存し、ベクトル化も完了しました。" if embedding else "新しいQ&Aを保存しました（ベクトル化には失敗）"
        return {"message": message, "fallback": result.data[0]}
    
    except HTTPException:
        raise # 400エラーなどをそのまま返す
    except Exception as e:
        # DB制約エラーのハンドリング
        if "23502" in str(e) and "category_name" in str(e):
             raise HTTPException(status_code=400, detail="category_name は必須です (DB Error 23502)")
        logging.error(f"Q&A作成エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# [main.py] 1190行目あたり

@app.put("/api/fallbacks/{qa_id}")
async def update_fallback(qa_id: int, request: Dict[str, Any], user: dict = Depends(require_auth)):
    """Q&Aを更新（テキスト変更時に自動でベクトル化）"""
    if not db_client or not settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")
    try:
        update_data = {}
        
        # 1. static_response (Q&Aテキスト) が更新されるかチェック
        if "static_response" in request:
            new_content = request["static_response"]
            update_data["static_response"] = new_content
            
            # --- 自動ベクトル化 ---
            embedding_model = settings_manager.settings.get("embedding_model", "text-embedding-004")
            logging.info(f"Q&A {qa_id} のテキストが変更されたため、ベクトルを再生成します...")
            try:
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=new_content
                )
                update_data["embedding"] = embedding_response["embedding"]
                logging.info(f"Q&A {qa_id} のベクトル再生成が完了しました。")
            except Exception as e:
                logging.error(f"Q&Aベクトル再生成エラー: {e}")
                update_data["embedding"] = None
                logging.warning(f"Q&A {qa_id} のベクトル化に失敗しましたが、テキストは更新します。")

        # 2. url_to_summarize も更新可能 (古いカラムが残っている場合)
        if "url_to_summarize" in request:
            update_data["url_to_summarize"] = request.get("url_to_summarize")
            
        # ★★★ 3. category_name の更新に対応 ★★★
        if "category_name" in request:
            new_category = request.get("category_name")
            if not new_category or not new_category.strip():
                 raise HTTPException(status_code=400, detail="category_name を空にすることはできません")
            update_data["category_name"] = new_category

        if not update_data:
            raise HTTPException(status_code=400, detail="更新するデータがありません")
        
        result = db_client.client.table("category_fallbacks").update(update_data).eq("id", qa_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Q&Aが見つかりません")
        
        logging.info(f"Q&A {qa_id} を更新しました（管理者: {user.get('email')}）")
        return {"message": "Q&Aを更新しました", "fallback": result.data[0]}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Q&A更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/fallbacks/{qa_id}")
async def delete_fallback(qa_id: int, user: dict = Depends(require_auth)):
    """Q&Aを削除"""
    if not db_client:
        raise HTTPException(503, "DB not initialized")
    try:
        result = db_client.client.table("category_fallbacks").delete().eq("id", qa_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="Q&Aが見つかりません")
        
        logging.info(f"Q&A {qa_id} を削除しました（管理者: {user.get('email')}）")
        return {"message": "Q&Aを削除しました"}
    except Exception as e:
        logging.error(f"Q&A削除エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/fallbacks/vectorize-all")
async def vectorize_all_missing_fallbacks(user: dict = Depends(require_auth)):
    """embedding が NULL のQ&Aをすべてベクトル化する"""
    if not db_client or not settings_manager:
        raise HTTPException(503, "DBまたは設定マネージャーが初期化されていません")

    logging.info(f"全Q&Aのベクトル化処理を開始...（管理者: {user.get('email')}）")
    
    try:
        # 1. embedding が NULL の Q&A をすべて取得
        response = db_client.client.table("category_fallbacks").select("id, static_response").is_("embedding", "null").execute()
        
        if not response.data:
            return {"message": "ベクトル化が必要なQ&Aはありませんでした。"}

        embedding_model = settings_manager.settings.get("embedding_model", "text-embedding-004")
        count = 0
        
        for item in response.data:
            item_id = item['id']
            text_to_vectorize = item['static_response']

            if not text_to_vectorize or not text_to_vectorize.strip():
                logging.warning(f"Q&A ID {item_id}: テキストが空のためスキップします。")
                continue

            try:
                # 2. ベクトル化
                embedding_response = genai.embed_content(
                    model=embedding_model,
                    content=text_to_vectorize
                )
                new_embedding = embedding_response["embedding"]

                # 3. DBを更新
                db_client.client.table("category_fallbacks").update({
                    "embedding": new_embedding
                }).eq("id", item_id).execute()
                
                logging.info(f"Q&A ID {item_id}: ベクトル化完了。")
                count += 1
                await asyncio.sleep(1) # APIレート制限対策

            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning(f"APIレート制限のため30秒待機します... (ID {item_id})")
                    await asyncio.sleep(30)
                    # 1回だけ再試行
                    embedding_response = genai.embed_content(model=embedding_model, content=text_to_vectorize)
                    new_embedding = embedding_response["embedding"]
                    db_client.client.table("category_fallbacks").update({"embedding": new_embedding}).eq("id", item_id).execute()
                    logging.info(f"Q&A ID {item_id}: (再試行) ベクトル化完了。")
                    count += 1
                else:
                    logging.error(f"Q&A ID {item_id} のベクトル化エラー: {e}")

        logging.info(f"全Q&Aベクトル化処理完了。 {count}件を処理しました。")
        return {"message": f"ベクトル化処理が完了しました。{count}件のQ&Aを更新しました。"}

    except Exception as e:
        logging.error(f"全Q&Aベクトル化処理中にエラーが発生: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feedback/stats")
async def get_feedback_stats():
    try:
        stats = feedback_manager.get_feedback_stats()
        return stats
    except Exception as e:
        logging.error(f"フィードバック統計取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 設定管理API ---
@app.post("/settings")
async def update_settings(settings: Settings):
    if not settings_manager:
        raise HTTPException(503, "設定マネージャーが初期化されていません")
    try:
        await settings_manager.update_settings(settings.dict(exclude_none=True))
        return {"message": "設定を更新しました"}
    except Exception as e:
        logging.error(f"設定更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- WebSocketエンドポイント ---
@app.websocket("/ws/settings")
async def websocket_endpoint(websocket: WebSocket):
    if not settings_manager:
        await websocket.close(code=1011, reason="Settings manager not initialized")
        return
    await settings_manager.add_websocket(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        settings_manager.remove_websocket(websocket)

# --------------------------------------------------------------------------
# 6. 開発用サーバー起動
# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)