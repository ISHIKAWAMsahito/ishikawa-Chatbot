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
from typing import List, Optional
from contextlib import asynccontextmanager

# --- サードパーティライブラリ ---
import google.generativeai as genai
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
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") # サービスキーを使用

# 定数
ACTIVE_COLLECTION_NAME = "student-knowledge-base"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JST = timezone(timedelta(hours=+9), 'JST')

# --------------------------------------------------------------------------
# 3. 内部コンポーネントの定義 (単一ファイル化)
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
            return " ".join(t.strip() for t in soup.stripped_strings)
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
    
    def save_log(self, log_id, query, response, context, category):
        """チャットログを保存"""
        try:
            file_exists = os.path.exists(self.logs_file)
            with open(self.logs_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['log_id', 'timestamp', 'query', 'response', 'context', 'category'])
                writer.writerow([log_id, datetime.now(JST).isoformat(), query, response, context, category])
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

# --------------------------------------------------------------------------
# 4. FastAPIアプリケーションのセットアップ
# --------------------------------------------------------------------------
db_client: Optional[SupabaseClientManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションのライフサイクル管理"""
    global db_client
    logging.info("--- アプリケーション起動処理開始 ---")
    
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
    rating: str  # "good" or "bad"

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
            embedding_response = genai.embed_content(model=embedding_model, content=chunk)
            embedding = embedding_response["embedding"]
            db_client.insert_document(chunk, embedding, metadata)
        
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
            embedding_response = genai.embed_content(model=req.embedding_model, content=chunk)
            embedding = embedding_response["embedding"]
            db_client.insert_document(chunk, embedding, metadata)
        
        return {"chunks": len(chunks)}
    except Exception as e: 
        logging.error(f"Scrape error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- チャットAPI ---
async def chat_streamer(query_data: ChatQuery):
    """チャット応答のストリーミング処理"""
    log_id = log_manager.generate_log_id()
    yield f"data: {json.dumps({'log_id': log_id})}\n\n"
    
    if not db_client: 
        yield f"data: {json.dumps({'content': 'データベースが初期化されていません。'})}\n\n"
        return
    
    try:
        # カテゴリ分類
        available_categories = db_client.get_distinct_categories(query_data.collection)
        if not available_categories:
            yield f"data: {json.dumps({'content': '現在参照できる知識がありません。'})}\n\n"
            return
        
        classification_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""利用可能なカテゴリ: {', '.join(available_categories)}
ユーザーの質問: 「{query_data.query}」

質問に最も関連するカテゴリ名を一つだけ出力してください。該当がなければ「その他」と出力。"""
        
        response = await classification_model.generate_content_async(prompt)
        classified_category = response.text.strip()
        
        message_content = f'（カテゴリ: {classified_category} を検索中...）\n'
        yield f"data: {json.dumps({'content': message_content})}\n\n"

        # 関連文書検索
        query_embedding_response = genai.embed_content(model=query_data.embedding_model, content=query_data.query)
        query_embedding = query_embedding_response["embedding"]
        
        search_results = db_client.search_documents(
            collection_name=query_data.collection, 
            category=classified_category,
            embedding=query_embedding, 
            match_count=query_data.top_k
        )
        
        context = "\n".join([doc['content'] for doc in search_results]) or "関連情報は見つかりませんでした。"

        # 最終回答生成
        final_model = genai.GenerativeModel(query_data.model)
        final_prompt = f"""参考情報:
---
{context}
---

質問: 「{query_data.query}」

参考情報に基づいて、学生からの質問に親切かつ丁寧に回答してください。参考情報に該当する内容がない場合は、一般的な回答を提供してください。"""
        
        stream = await final_model.generate_content_async(final_prompt, stream=True)
        
        full_response_text = ""
        async for chunk in stream:
            full_response_text += chunk.text
            yield f"data: {json.dumps({'content': chunk.text})}\n\n"
        
        # ログ保存
        log_manager.save_log(log_id, query_data.query, full_response_text, context, classified_category)
        
    except Exception as e:
        logging.error(f"チャット処理エラー: {e}\n{traceback.format_exc()}")
        yield f"data: {json.dumps({'content': f'エラーが発生しました: {e}'})}\n\n"

@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    """管理者用チャットエンドポイント"""
    return StreamingResponse(chat_streamer(query), media_type="text/event-stream")

@app.post("/chat_for_client")
async def chat_for_client(query: ClientChatQuery):
    """クライアント用チャットエンドポイント（固定設定）"""
    chat_query = ChatQuery(
        query=query.query,
        model="gemini-1.5-flash-latest",
        embedding_model="text-embedding-004",
        top_k=5,
        collection=ACTIVE_COLLECTION_NAME
    )
    return StreamingResponse(chat_streamer(chat_query), media_type="text/event-stream")

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
    try:
        # 設定をブロードキャスト
        await manager.broadcast(json.dumps({
            "type": "settings_update",
            "data": settings.dict(exclude_none=True)
        }))
        return {"message": "設定を更新しました"}
    except Exception as e:
        logging.error(f"設定更新エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- WebSocketエンドポイント ---
@app.websocket("/ws/settings")
async def websocket_endpoint(websocket: WebSocket):
    """設定同期用WebSocket"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # クライアントからのメッセージは特に処理しない
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --------------------------------------------------------------------------
# 6. 開発用サーバー起動
# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)