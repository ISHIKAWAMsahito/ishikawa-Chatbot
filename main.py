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
# [重要] requirements.txt に全てのライブラリが含まれていることを確認してください
import google.generativeai as genai
from dotenv import load_dotenv

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, WebSocket
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
if not GEMINI_API_KEY: raise ValueError("環境変数 'GEMINI_API_KEY' が設定されていません。")
genai.configure(api_key=GEMINI_API_KEY)

# Auth0設定
AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")

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
    # (実装は元のファイルから流用)
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        chunks.append(text[start:end])
        start += max_length - overlap
    return chunks

class DocumentProcessor:
    # (実装は元のファイルから流用)
    def process(self, filename: str, content: bytes) -> List[str]:
        text = ""
        try:
            if filename.lower().endswith(".pdf"):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                for page in pdf_reader.pages: text += page.extract_text() or ""
            elif filename.lower().endswith(".docx"):
                doc = DocxDocument(io.BytesIO(content))
                for para in doc.paragraphs: text += para.text + "\n"
            else: text = content.decode('utf-8', errors='ignore')
        except Exception as e:
            logging.error(f"ファイル処理エラー ({filename}): {e}")
            return []
        return split_text(text)

class WebScraper:
    # (実装は元のファイルから流用)
    def scrape(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(["script", "style", "header", "footer", "nav", "aside"]): element.decompose()
            return " ".join(t.strip() for t in soup.stripped_strings)
        except Exception as e:
            logging.error(f"スクレイピングエラー ({url}): {e}")
            return ""

class LogManager:
    # (実装は元のファイルから流用)
    def generate_log_id(self) -> str: return str(uuid.uuid4())
    def save_log(self, log_id, query, response, context, category):
        logging.info(f"LOG [{log_id}]: Q='{query}' A='{response[:80].strip()}...' Cat='{category}'")

class SupabaseClientManager:
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def get_db_type(self) -> str: return "supabase"

    def insert_document(self, content: str, embedding: List[float], metadata: dict):
        self.client.table("documents").insert({
            "content": content, "embedding": embedding, "metadata": metadata
        }).execute()

    def search_documents(self, collection_name: str, category: str, embedding: List[float], match_count: int) -> List[dict]:
        params = {
            "p_collection_name": collection_name, "p_category": category,
            "p_query_embedding": embedding, "p_match_count": match_count
        }
        return self.client.rpc("match_documents", params).execute().data

    def get_documents_by_collection(self, collection_name: str) -> List[dict]:
        return self.client.table("documents").select("id, metadata").eq("metadata->>collection_name", collection_name).execute().data
    
    def count_chunks_in_collection(self, collection_name: str) -> int:
        return self.client.table("documents").select("id", count='exact').eq("metadata->>collection_name", collection_name).execute().count

    def get_distinct_categories(self, collection_name: str) -> List[str]:
        try:
            result = self.client.rpc("get_distinct_categories", {"p_collection_name": collection_name}).execute()
            return [item['category'] for item in result.data if item.get('category')]
        except Exception as e:
            logging.error(f"RPC 'get_distinct_categories' の呼び出しエラー: {e}")
            return ["その他"]

# --------------------------------------------------------------------------
# 4. FastAPIアプリケーションのセットアップ
# --------------------------------------------------------------------------
db_client: Optional[SupabaseClientManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_client
    logging.info("--- アプリケーション起動処理開始 ---")
    if SUPABASE_URL and SUPABASE_KEY:
        db_client = SupabaseClientManager(url=SUPABASE_URL, key=SUPABASE_KEY)
        logging.info("Supabaseクライアントの初期化完了。")
    else:
        logging.warning("Supabaseの環境変数が設定されていません。")
    yield
    logging.info("--- アプリケーション終了処理 ---")

app = FastAPI(lifespan=lifespan)
if APP_SECRET_KEY: app.add_middleware(SessionMiddleware, secret_key=APP_SECRET_KEY)
else: logging.warning("APP_SECRET_KEYが未設定のため、セッション機能（ログイン）は動作しません。")

oauth = OAuth()
if all([AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN]):
    oauth.register(
        name='auth0', client_id=AUTH0_CLIENT_ID, client_secret=AUTH0_CLIENT_SECRET,
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
    query: str; model: str; embedding_model: str; top_k: int
    collection: str = ACTIVE_COLLECTION_NAME # 固定値

class ScrapeRequest(BaseModel):
    url: str; category: str; embedding_model: str
    collection_name: str = ACTIVE_COLLECTION_NAME # 固定値

# --------------------------------------------------------------------------
# 5. APIエンドポイント定義
# --------------------------------------------------------------------------

# --- 認証とHTML提供 ---
def require_auth(request: Request):
    user = request.session.get('user')
    if not user: raise HTTPException(status_code=307, headers={'Location': '/login'})
    return user

@app.get('/login')
async def login(request: Request):
    if 'auth0' not in oauth._clients: raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    return await oauth.auth0.authorize_redirect(request, request.url_for('auth'))

@app.get('/auth')
async def auth(request: Request):
    if 'auth0' not in oauth._clients: raise HTTPException(status_code=500, detail="Auth0 is not configured.")
    token = await oauth.auth0.authorize_access_token(request)
    if userinfo := token.get('userinfo'): request.session['user'] = dict(userinfo)
    return RedirectResponse(url='/admin')

@app.get('/logout')
async def logout(request: Request):
    request.session.pop('user', None)
    if not all([AUTH0_DOMAIN, AUTH0_CLIENT_ID]): return RedirectResponse(url='/')
    return RedirectResponse(f"https://{AUTH0_DOMAIN}/v2/logout?returnTo={request.url_for('serve_client')}&client_id={AUTH0_CLIENT_ID}")

@app.get("/", response_class=FileResponse)
async def serve_client(): return FileResponse(os.path.join(BASE_DIR, "client.html"))
@app.get("/admin", response_class=FileResponse, dependencies=[Depends(require_auth)])
async def serve_admin(): return FileResponse(os.path.join(BASE_DIR, "admin.html"))
@app.get("/favicon.ico", include_in_schema=False)
async def favicon(): return Response(status_code=204)

# --- ステータス確認 ---
@app.get("/health")
async def health_check(): return {"status": "ok", "database": db_client.get_db_type() if db_client else "uninitialized"}
@app.get("/gemini/status")
async def gemini_status():
    if not GEMINI_API_KEY: return {"connected": False}
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        return {"connected": True, "models": models}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# --- ナレッジ管理API ---
@app.get("/collections/{collection_name}/documents", dependencies=[Depends(require_auth)])
async def get_documents(collection_name: str):
    if not db_client: raise HTTPException(503, "DB not initialized")
    return {
        "documents": db_client.get_documents_by_collection(collection_name),
        "count": db_client.count_chunks_in_collection(collection_name)
    }

@app.post("/upload", dependencies=[Depends(require_auth)])
async def upload_document(file: UploadFile = File(...), embedding_model: str = Form(...)):
    if not db_client: raise HTTPException(503, "DB not initialized")
    try:
        category = file.filename.split('_')[0] if '_' in file.filename else "未分類"
        content = await file.read()
        chunks = document_processor.process(file.filename, content)
        for chunk in chunks:
            metadata = {"source": file.filename, "collection_name": ACTIVE_COLLECTION_NAME, "category": category}
            embedding = genai.embed_content(model=embedding_model, content=chunk)["embedding"]
            db_client.insert_document(chunk, embedding, metadata)
        return {"chunks": len(chunks)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/scrape", dependencies=[Depends(require_auth)])
async def scrape_website(req: ScrapeRequest):
    if not db_client: raise HTTPException(503, "DB not initialized")
    try:
        content = web_scraper.scrape(req.url)
        if not content: raise HTTPException(status_code=400, detail="コンテンツを取得できませんでした。")
        chunks = split_text(content)
        for chunk in chunks:
            metadata = {"source": req.url, "collection_name": req.collection_name, "category": req.category}
            embedding = genai.embed_content(model=req.embedding_model, content=chunk)["embedding"]
            db_client.insert_document(chunk, embedding, metadata)
        return {"chunks": len(chunks)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

# --- チャットAPI ---
async def chat_streamer(query_data: ChatQuery):
    log_id = log_manager.generate_log_id()
    yield f"data: {json.dumps({'log_id': log_id})}\n\n"
    if not db_client: 
        yield f"data: {json.dumps({'content': 'データベースが初期化されていません。'})}\n\n"
        return
    try:
        available_categories = db_client.get_distinct_categories(query_data.collection)
        if not available_categories:
            yield f"data: {json.dumps({'content': '現在参照できる知識がありません。'})}\n\n"
            return
        
        classification_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"利用可能なカテゴリ: {', '.join(available_categories)}\nユーザーの質問: 「{query_data.query}」\n質問に最も関連するカテゴリ名を一つだけ出力してください。該当がなければ「その他」と出力。"
        response = await classification_model.generate_content_async(prompt)
        classified_category = response.text.strip()
        
        message_content = f'（カテゴリ: {classified_category} を検索中...）\n'
        yield f"data: {json.dumps({'content': message_content})}\n\n"

        query_embedding = genai.embed_content(model=query_data.embedding_model, content=query_data.query)["embedding"]
        search_results = db_client.search_documents(
            collection_name=query_data.collection, category=classified_category,
            embedding=query_embedding, match_count=query_data.top_k
        )
        context = "\n".join([doc['content'] for doc in search_results]) or "関連情報は見つかりませんでした。"

        final_model = genai.GenerativeModel(query_data.model)
        final_prompt = f"参考情報:\n---\n{context}\n---\n質問: 「{query_data.query}」\n\n参考情報に基づいて、学生からの質問に親切かつ丁寧に回答してください。"
        stream = await final_model.generate_content_async(final_prompt, stream=True)
        
        full_response_text = ""
        async for chunk in stream:
            full_response_text += chunk.text
            yield f"data: {json.dumps({'content': chunk.text})}\n\n"
        
        log_manager.save_log(log_id, query_data.query, full_response_text, context, classified_category)
    except Exception as e:
        logging.error(f"チャット処理エラー: {e}\n{traceback.format_exc()}")
        yield f"data: {json.dumps({'content': f'エラーが発生しました: {e}'})}\n\n"

@app.post("/chat")
async def chat_endpoint(query: ChatQuery):
    return StreamingResponse(chat_streamer(query), media_type="text/event-stream")

# --------------------------------------------------------------------------
# 6. 開発用サーバー起動
# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

