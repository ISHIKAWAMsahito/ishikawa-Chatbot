# --------------------------------------------------------------------------
# 1. ライブラリのインポート
# --------------------------------------------------------------------------
import os
import json
import uvicorn
import traceback
import csv
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

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse, FileResponse, Response
from pydantic import BaseModel

from supabase import create_client, Client
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document as DocxDocument

from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth

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

# Auth0設定
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

# キーワードマッピング
KEYWORD_MAP = {
    "授業": ["学事暦", "シラバス", "授業計画", "年間スケジュール", "夏休み", "冬休み", "春休み", "夏季休業", "冬季休業", "春季休業"],
    "カリキュラム": ["カリキュラム", "履修", "科目", "単位", "授業", "必修", "選択"],
    "欠席手続き": ["忌引き", "休む", "欠席", "休講", "病気", "インフルエンザ", "欠席届", "公認欠席", "特別欠席"],
    "試験": ["試験", "テスト", "追試", "再試", "成績", "レポート", "課題"],
    "証明書": ["証明書", "学割", "発行", "学生証", "学籍", "休学", "復学", "退学", "再入学", "転学部", "転学科", "卒業", "在学", "卒業見込"],
    "留学": ["留学", "海外", "協定校", "IEC", "国際交流"],
    "学生支援": ["サークル", "部活", "ボランティア", "障がい", "障害"],
    "経済支援": ["奨学金", "授業料", "学費", "免除", "支援金", "学費支援", "家計急変", "被災学生", "授業料減免"],
    "資格": ["資格", "免許", "講座", "キャリア"],
    "相談": ["相談", "カウンセリング", "悩み", "メンタル", "ハラスメント", "トラブル"],
    "施設": ["図書館", "食堂", "購買", "場所", "どこ", "Wi-Fi", "PC", "パソコン", "教室", "体育館", "グラウンド", "駐車場"],
    "生協": ["生協", "コープ", "教科書", "共済", "組合員", "購買", "食堂", "書籍", "パソコン", "カフェテリア", "学内ショップ"],
}

# カテゴリ情報
CATEGORY_INFO = {
    "授業": {
        "url_to_summarize": "https://www.sgu.ac.jp/information/schedule.html",
        "static_response": "学事暦や年間スケジュールに関するご質問ですね。\n- **夏期休業期間（夏休み）**: 8月上旬～9月中旬\n- **冬季休業期間（冬休み）**: 12月下旬～1月上旬\n- **春季休業期間（春休み）**: 2月上旬～3月下旬\n正確な日付は、https://www.sgu.ac.jp/information/schedule.html でご確認ください。"
    },
    "証明書": {
        "url_to_summarize": "https://www.sgu.ac.jp/campuslife/support/certification.html",
        "static_response": "各種証明書の発行については、https://www.sgu.ac.jp/campuslife/support/certification.html のページをご確認ください。"
    },
    "経済支援": {
        "url_to_summarize": "https://www.sgu.ac.jp/campuslife/scholarship/",
        "static_response": "奨学金や授業料減免については、https://www.sgu.ac.jp/campuslife/scholarship/ や https://www.sgu.ac.jp/tuition/j09tjo00000f665g.html のページをご確認ください。"
    },
}


# --------------------------------------------------------------------------
# 3. 内部コンポーネントの定義
# --------------------------------------------------------------------------

def split_text(text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
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
    # マークダウン形式のリンク [テキスト](URL) を HTML に変換
    md_link_pattern = r'\[([^\]]+)\]\((https?://[^\s\)]+)\)'
    text = re.sub(md_link_pattern, r'<a href="\2" target="_blank">\1</a>', text)
    
    # 単体のURLをリンクに変換（マークダウンリンクでないもの）
    url_pattern = r'(?<!\]\()(https?://[^\s\[\]()<>]+)(?!\))'
    text = re.sub(url_pattern, r'<a href="\1" target="_blank">\1</a>', text)
    
    return text

# レート制限対応のヘルパー関数
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
                        temperature=0.7
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
                    # エラーメッセージから待機時間を抽出
                    wait_time = 15  # デフォルト15秒
                    if "retry in" in error_str:
                        try:
                            import re
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
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(["script", "style", "header", "footer", "nav", "aside"]): 
                element.decompose()
            text = " ".join(t.strip() for t in soup.stripped_strings)
            return re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            logging.error(f"スクレイピングエラー ({url}): {e}")
            return ""

class LogManager:
    """ログ管理クラス"""
    def __init__(self):
        self.logs_file = "chat_logs.csv"
        self.feedback_file = "feedback.csv"
        
    def generate_log_id(self) -> str: 
        return str(uuid.uuid4())
    
    def save_log(self, log_id, query, response, context, category, has_specific_info=False):
        """チャットログを保存"""
        try:
            file_exists = os.path.exists(self.logs_file)
            with open(self.logs_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['log_id', 'timestamp', 'query', 'response', 'context', 'category', 'has_specific_info'])
                writer.writerow([log_id, datetime.now(JST).isoformat(), query, response, context, category, has_specific_info])
        except Exception as e:
            logging.error(f"ログ保存エラー: {e}")
        
        logging.info(f"LOG [{log_id}]: Q='{query}' A='{response[:80].strip()}...' Cat='{category}'")

    def save_feedback(self, log_id: str, rating: str):
        """フィードバックを保存"""
        try:
            file_exists = os.path.exists(self.feedback_file)
            with open(self.feedback_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['log_id', 'timestamp', 'rating'])
                writer.writerow([log_id, datetime.now(JST).isoformat(), rating])
        except Exception as e:
            logging.error(f"フィードバック保存エラー: {e}")

    def get_logs_with_feedback(self) -> List[dict]:
        """ログとフィードバックを結合して返す"""
        logs = []
        feedback_dict = {}
        
        # フィードバックを読み込み
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        feedback_dict[row['log_id']] = row['rating']
            except Exception as e:
                logging.error(f"フィードバック読み込みエラー: {e}")
        
        # ログを読み込み
        if os.path.exists(self.logs_file):
            try:
                with open(self.logs_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        log_entry = dict(row)
                        log_entry['rating'] = feedback_dict.get(row['log_id'])
                        logs.append(log_entry)
            except Exception as e:
                logging.error(f"ログ読み込みエラー: {e}")
        
        return logs

class SupabaseClientManager:
    """Supabaseクライアント管理クラス"""
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def get_db_type(self) -> str: 
        return "supabase"

    def insert_document(self, content: str, embedding: List[float], metadata: dict):
        """ドキュメントを挿入"""
        self.client.table("documents").insert({
            "content": content, 
            "embedding": embedding, 
            "metadata": metadata
        }).execute()

    def search_documents(self, collection_name: str, category: str, embedding: List[float], match_count: int) -> List[dict]:
        """ドキュメント検索"""
        params = {
            "p_collection_name": collection_name, 
            "p_category": category,
            "p_query_embedding": embedding, 
            "p_match_count": match_count
        }
        result = self.client.rpc("match_documents", params).execute()
        return result.data or []

    def get_documents_by_collection(self, collection_name: str) -> List[dict]:
        """コレクション内のドキュメント一覧を取得"""
        result = self.client.table("documents").select("id, metadata").eq("metadata->>collection_name", collection_name).execute()
        return result.data or []
    
    def count_chunks_in_collection(self, collection_name: str) -> int:
        """コレクション内のチャンク数をカウント"""
        result = self.client.table("documents").select("id", count='exact').eq("metadata->>collection_name", collection_name).execute()
        return result.count or 0

    def get_distinct_categories(self, collection_name: str) -> List[str]:
        """コレクション内のカテゴリ一覧を取得"""
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
            "model": "gemini-1.5-flash-latest", 
            "collection": ACTIVE_COLLECTION_NAME, 
            "embedding_model": "text-embedding-004", 
            "top_k": 5
        }
        self.websocket_connections: List[WebSocket] = []
        self.settings_file = os.path.join(BASE_DIR, "shared_settings.json")
        self.load_settings()

    def load_settings(self):
        """設定ファイルから読み込み"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    self.settings.update(json.load(f))
        except Exception as e:
            logging.error(f"設定ファイルの読み込みエラー: {e}")

    def save_settings(self):
        """設定ファイルに保存"""
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"設定ファイルの保存エラー: {e}")

    async def update_settings(self, new_settings: Dict[str, Any]):
        """設定を更新してWebSocketでブロードキャスト"""
        self.settings.update(new_settings)
        self.save_settings()
        await self.broadcast_settings()

    async def add_websocket(self, websocket: WebSocket):
        """WebSocket接続を追加"""
        await websocket.accept()
        self.websocket_connections.append(websocket)

    def remove_websocket(self, websocket: WebSocket):
        """WebSocket接続を削除"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)

    async def broadcast_settings(self):
        """設定をすべてのWebSocket接続にブロードキャスト"""
        message = {"type": "settings_update", "data": self.settings}
        disconnected = []
        
        for conn in self.websocket_connections:
            try:
                await conn.send_json(message)
            except:
                disconnected.append(conn)
        
        for conn in disconnected:
            self.remove_websocket(conn)

async def create_tiered_response(category: str, model: str) -> str:
    """カテゴリに基づいてステージ応答を作成"""
    info = CATEGORY_INFO.get(category)
    if not info:
        return "申し訳ありませんが、お尋ねの件について情報が見つかりませんでした。[大学公式サイト](https://www.sgu.ac.jp/)をご確認ください。"
    
    # URLから要約を試みる（エラー時は静的応答を使用）
    if 'url_to_summarize' in info:
        try:
            web_scraper = WebScraper()
            content = web_scraper.scrape(info['url_to_summarize'])
            if content and len(content) > 100:
                prompt = f"以下の大学公式サイトの内容を学生向けに要約してください：\n{content[:4000]}"
                gemini_model = genai.GenerativeModel(model)
                response = await safe_generate_content(gemini_model, prompt, stream=False)
                return f"**▼ {category}に関する公式情報**\n{response.text}\n\n詳細は[こちら]({info['url_to_summarize']})をご確認ください。"
        except Exception as e:
            logging.warning(f"URL要約エラー: {e}")
    
    return info.get("static_response", "情報が見つかりませんでした。")

# --------------------------------------------------------------------------
# 4. FastAPIアプリケーションのセットアップ
# --------------------------------------------------------------------------
db_client: Optional[SupabaseClientManager] = None
settings_manager: Optional[SettingsManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    global db_client, settings_manager
    logging.info("--- アプリケーション起動処理開始 ---")
    
    # 設定マネージャー初期化
    settings_manager = SettingsManager()
    
    # Supabase初期化
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            db_client = SupabaseClientManager(url=SUPABASE_URL, key=SUPABASE_KEY)
            logging.info("Supabaseクライアントの初期化完了。")
        except Exception as e:
            logging.error(f"Supabase初期化エラー: {e}")
    else:
        logging.warning("Supabaseの環境変数が設定されていません。")
    
    yield
    logging.info("--- アプリケーション終了処理 ---")

app = FastAPI(lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=APP_SECRET_KEY)

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
    logging.warning("Auth0の設定が不完全なため、認証機能は動作しません。")

# --- グローバルインスタンス ---
log_manager = LogManager()
document_processor = DocumentProcessor()
web_scraper = WebScraper()

# --- データモデル定義 ---
class ChatQuery(BaseModel):
    query: str
    model: str = "gemini-1.5-flash-latest"
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
    log_id: str
    rating: str

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

# --- 認証関数 ---
def require_auth(request: Request):
    """認証が必要なエンドポイント用の依存関数"""
    user = request.session.get('user')
    if not user and all([AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN]):
        raise HTTPException(status_code=307, headers={'Location': '/login'})
    return user

# --- 認証とHTML提供 ---
@app.get('/login')
async def login(request: Request):
    if 'auth0' not in oauth._clients: 
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    return await oauth.auth0.authorize_redirect(request, request.url_for('auth'))

@app.get('/auth')
async def auth(request: Request):
    if 'auth0' not in oauth._clients: 
        raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    token = await oauth.auth0.authorize_access_token(request)
    if userinfo := token.get('userinfo'): 
        request.session['user'] = dict(userinfo)
    return RedirectResponse(url='/admin')

@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    if not all([AUTH0_DOMAIN, AUTH0_CLIENT_ID]): 
        return RedirectResponse(url='/')
    return RedirectResponse(f"https://{AUTH0_DOMAIN}/v2/logout?returnTo={request.url_for('serve_client')}&client_id={AUTH0_CLIENT_ID}")

@app.get("/", response_class=FileResponse)
async def serve_client(): 
    return FileResponse(os.path.join(BASE_DIR, "client.html"))

@app.get("/admin", response_class=FileResponse)
async def serve_admin(request: Request):
    # 認証設定がある場合は認証をチェック
    if all([AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN]):
        require_auth(request)
    return FileResponse(os.path.join(BASE_DIR, "admin.html"))

@app.get("/log", response_class=FileResponse)
async def serve_log(request: Request):
    # 認証設定がある場合は認証をチェック
    if all([AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN]):
        require_auth(request)
    return FileResponse(os.path.join(BASE_DIR, "log.html"))

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

# --- コレクション管理API ---
@app.get("/collections")
async def get_collections():
    """コレクション一覧を返す（固定値）"""
    return [{"name": ACTIVE_COLLECTION_NAME, "count": db_client.count_chunks_in_collection(ACTIVE_COLLECTION_NAME) if db_client else 0}]

@app.post("/collections")
async def create_collection(request: dict):
    """コレクション作成（固定値のため何もしない）"""
    return {"message": f"コレクション「{ACTIVE_COLLECTION_NAME}」は既に存在しています"}

@app.delete("/collections/{collection_name}")
async def delete_collection(collection_name: str):
    """コレクション削除（固定値のため削除不可）"""
    if collection_name == ACTIVE_COLLECTION_NAME:
        raise HTTPException(status_code=400, detail="このコレクションは削除できません")
    return {"message": "コレクションが見つかりません"}

# --- ナレッジ管理API ---
@app.get("/collections/{collection_name}/documents")
async def get_documents(collection_name: str):
    if not db_client: 
        raise HTTPException(503, "DB not initialized")
    return {
        "documents": db_client.get_documents_by_collection(collection_name),
        "count": db_client.count_chunks_in_collection(collection_name)
    }

@app.post("/upload")
async def upload_document(
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
            # レート制限対応でembeddingを生成
            try:
                embedding_response = genai.embed_content(model=embedding_model, content=chunk)
                embedding = embedding_response["embedding"]
                db_client.insert_document(chunk, embedding, metadata)
                # API制限を考慮して少し待機
                await asyncio.sleep(1)
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    logging.warning("埋め込み生成でAPI制限に達しました。30秒待機します。")
                    await asyncio.sleep(30)
                    # 再試行
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
            # レート制限対応
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
async def enhanced_chat_logic(request: Request, chat_req: ChatQuery):
    """統合チャットロジック（RAG対応）"""
    user_input = chat_req.query.strip()
    log_id = log_manager.generate_log_id()
    
    yield f"data: {json.dumps({'log_id': log_id})}\n\n"

    try:
        # --- 通常の学内情報FAQ処理 ---
        if not all([db_client, GEMINI_API_KEY]):
            error_msg = "システムが利用できません。管理者にお問い合わせください。"
            log_manager.save_log(log_id, user_input, error_msg, "", "エラー", False)
            yield f"data: {json.dumps({'content': error_msg})}\n\n"
            return

        # カテゴリ分類
        category = next((cat for cat, keys in KEYWORD_MAP.items() if any(key in user_input for key in keys)), "その他")
        
        full_response = ""
        context = ""
        has_specific_info = False

        try:
            # データベースから関連情報を検索
            query_embedding_response = genai.embed_content(model=chat_req.embedding_model, content=user_input)
            query_embedding = query_embedding_response["embedding"]
            
            search_results = db_client.search_documents(
                collection_name=chat_req.collection, 
                category=category,
                embedding=query_embedding, 
                match_count=chat_req.top_k
            )
            
            if search_results:
                context = "\n".join([doc['content'] for doc in search_results])
                has_specific_info = True
            
        except Exception as e:
            logging.error(f"データベース検索エラー: {e}")

        if has_specific_info:
            # データベースに情報がある場合
            prompt = f"""あなたは札幌学院大学の学生を支援するAIです。以下の情報を元に、質問に日本語で回答してください。

参考情報:
{context}

質問: {user_input}

回答:"""
            
            model = genai.GenerativeModel(chat_req.model)
            stream = await safe_generate_content(model, prompt, stream=True)
            
            async for chunk in stream:
                if chunk.text:
                    full_response += chunk.text
                    yield f"data: {json.dumps({'content': chunk.text})}\n\n"
        
        else:
            # データベースに情報がない場合、階層化応答
            tiered_response = await create_tiered_response(category, chat_req.model)
            full_response = tiered_response
            yield f"data: {json.dumps({'content': full_response})}\n\n"

        # URLをリンクに変換
        full_response = format_urls_as_links(full_response)
        
        # ログ保存
        log_manager.save_log(log_id, user_input, full_response, context, category, has_specific_info)

    except Exception as e:
        error_message = f"エラーが発生しました: {str(e)}"
        logging.error(f"チャットロジックエラー: {e}\n{traceback.format_exc()}")
        log_manager.save_log(log_id, user_input, error_message, "", "エラー", False)
        yield f"data: {json.dumps({'content': error_message})}\n\n"

@app.post("/chat")
async def chat_endpoint(request: Request, query: ChatQuery):
    """管理者用チャットエンドポイント"""
    return StreamingResponse(enhanced_chat_logic(request, query), media_type="text/event-stream")

@app.post("/chat_for_client")
async def chat_for_client(request: Request, query: ClientChatQuery):
    """クライアント用チャットエンドポイント（固定設定）"""
    if not settings_manager:
        raise HTTPException(503, "設定マネージャーが初期化されていません")
    
    chat_query = ChatQuery(
        query=query.query,
        model=settings_manager.settings.get("model", "gemini-1.5-flash-latest"),
        embedding_model=settings_manager.settings.get("embedding_model", "text-embedding-004"),
        top_k=settings_manager.settings.get("top_k", 5),
        collection=settings_manager.settings.get("collection", ACTIVE_COLLECTION_NAME)
    )
    return StreamingResponse(enhanced_chat_logic(request, chat_query), media_type="text/event-stream")

# --- フィードバックAPI ---
@app.post("/feedback")
async def save_feedback(feedback: FeedbackRequest):
    """フィードバック保存"""
    try:
        log_manager.save_feedback(feedback.log_id, feedback.rating)
        return {"message": "フィードバックを保存しました"}
    except Exception as e:
        logging.error(f"フィードバック保存エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- ログ取得API ---
@app.get("/logs")
async def get_logs():
    """チャットログ取得"""
    try:
        logs = log_manager.get_logs_with_feedback()
        return logs
    except Exception as e:
        logging.error(f"ログ取得エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 設定管理API ---
@app.post("/settings")
async def update_settings(settings: Settings):
    """設定更新（WebSocket経由で通知）"""
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
    """設定同期用WebSocket"""
    if not settings_manager:
        await websocket.close(code=1011, reason="Settings manager not initialized")
        return
    
    await settings_manager.add_websocket(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # クライアントからのメッセージは特に処理しない
    except WebSocketDisconnect:
        settings_manager.remove_websocket(websocket)

# --------------------------------------------------------------------------
# 6. 開発用サーバー起動
# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)